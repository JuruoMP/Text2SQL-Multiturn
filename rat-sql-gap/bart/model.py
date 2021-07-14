import os
import torch
import torch.nn as nn
import pytorch_lightning as pl

from configuration_bart import BartConfig
from modeling_bart import BartPretrainedModel, BartModel, BartForConditionalGeneration, shift_tokens_right
from evaluation import evaluate as evaluate_sparc


# class SQLBartModel(BartPretrainedModel):
#     def __init__(self, name_or_path):
#         self.config = BartConfig.from_pretrained(name_or_path)
#         super().__init__(self.config)
#         self.model = BartModel.from_pretrained(name_or_path, cache_dir='bart/cache')
#         self.register_buffer("final_logits_bias", torch.zeros((1, self.model.shared.num_embeddings)))
#         self.lm_head = nn.Linear(self.config.d_model, self.model.shared.num_embeddings, bias=False)
#
#     def get_encoder(self):
#         return self.model.get_encoder()
#
#     def get_decoder(self):
#         return self.model.get_decoder()
#
#     def forward(self, x):
#         bart_outputs = self.model(
#             input_ids=x['input_ids'],
#             attention_mask=x['attention_mask'],
#             decoder_input_ids=x['decoder_input_ids'],
#             decoder_attention_mask=x['decoder_attention_mask']
#         )
#         lm_logits = self.lm_head(bart_outputs[0]) + self.final_logits_bias
#         return lm_logits
#
#     def resize_token_embeddings(self, new_num_tokens: int) -> nn.Embedding:
#         new_embeddings = super().resize_token_embeddings(new_num_tokens)
#         self._resize_final_logits_bias(new_num_tokens)
#         return new_embeddings
#
#     def _resize_final_logits_bias(self, new_num_tokens: int) -> None:
#         old_num_tokens = self.final_logits_bias.shape[-1]
#         if new_num_tokens <= old_num_tokens:
#             new_bias = self.final_logits_bias[:, :new_num_tokens]
#         else:
#             extra_bias = torch.zeros((1, new_num_tokens - old_num_tokens), device=self.final_logits_bias.device)
#             new_bias = torch.cat([self.final_logits_bias, extra_bias], dim=1)
#         self.register_buffer("final_logits_bias", new_bias)
#
#     def get_output_embeddings(self):
#         return self.lm_head
#
#     def set_output_embeddings(self, new_embeddings):
#         self.lm_head = new_embeddings
#
#     def prepare_inputs_for_generation(
#         self,
#         decoder_input_ids,
#         past=None,
#         attention_mask=None,
#         head_mask=None,
#         decoder_head_mask=None,
#         cross_attn_head_mask=None,
#         use_cache=None,
#         encoder_outputs=None,
#         **kwargs
#     ):
#         # cut decoder_input_ids if past is used
#         if past is not None:
#             decoder_input_ids = decoder_input_ids[:, -1:]
#
#         return {
#             "input_ids": None,  # encoder_outputs is defined. input_ids not needed
#             "encoder_outputs": encoder_outputs,
#             "past_key_values": past,
#             "decoder_input_ids": decoder_input_ids,
#             "attention_mask": attention_mask,
#             "head_mask": head_mask,
#             "decoder_head_mask": decoder_head_mask,
#             "cross_attn_head_mask": cross_attn_head_mask,
#             "use_cache": use_cache,  # change this to avoid caching (presumably for debugging)
#         }
#
#     def prepare_decoder_input_ids_from_labels(self, labels: torch.Tensor):
#         return shift_tokens_right(labels, self.config.pad_token_id, self.config.decoder_start_token_id)
#
#     @staticmethod
#     def _reorder_cache(past, beam_idx):
#         reordered_past = ()
#         for layer_past in past:
#             # cached cross_attention states don't have to be reordered -> they are always the same
#             reordered_past += (
#                 tuple(past_state.index_select(0, beam_idx) for past_state in layer_past[:2]) + layer_past[2:],
#             )
#         return reordered_past


class SQLBartModel(BartForConditionalGeneration):
    def __init__(self, name_or_path):
        config = BartConfig.from_pretrained(name_or_path)
        super().__init__(config)
        self.model.load_state_dict(BartModel.from_pretrained(name_or_path).state_dict())


class SQLBart(pl.LightningModule):
    def __init__(self, tokenizer):
        super().__init__()
        self.tokenizer = tokenizer
        self.model = SQLBartModel(tokenizer.name_or_path)
        self.model.resize_token_embeddings(len(self.tokenizer))
        self.loss_fct = torch.nn.CrossEntropyLoss(ignore_index=-100)
        self.learning_rate = 1e-5
        self.check_interval = 1

    def forward(self, x):
        lm_logits = self.model(x)
        pred_ids = lm_logits.argmax(dim=-1)
        pred_str = [self.tokenizer.convert_ids_to_tokens(xx) for xx in pred_ids]
        masked_lm_loss = self.loss_fct(lm_logits.view(-1, len(self.tokenizer)), x['labels'].view(-1))
        return {'lm_logits': lm_logits, 'sql': pred_str, 'loss': masked_lm_loss}

    def training_step(self, x, batch_idx):
        masked_lm_loss = self.model(
            input_ids=x['input_ids'],
            attention_mask=x['attention_mask'],
            decoder_input_ids=x['decoder_input_ids'],
            decoder_attention_mask=x['decoder_attention_mask'],
            labels=x['labels']
        ).loss
        self.log('train_loss', masked_lm_loss, sync_dist=True)
        return masked_lm_loss

    def validation_step(self, x, batch_idx):
        masked_lm_loss = self.model(
            input_ids=x['input_ids'],
            attention_mask=x['attention_mask'],
            decoder_input_ids=x['decoder_input_ids'],
            decoder_attention_mask=x['decoder_attention_mask'],
            labels=x['labels']
        ).loss

        pred_lfs = []
        pred_ids = self.model.generate(x['input_ids'], num_beams=4, early_stopping=True, no_repeat_ngram_size=0)[:, 1:]
        for i in range(x['id'].size(0)):
            pred_lf = self.tokenizer.convert_ids_to_tokens(pred_ids[i])
            if self.tokenizer.eos_token in pred_lf:
                pred_lf = pred_lf[1:pred_lf.index(self.tokenizer.eos_token)]
            else:
                pred_lf = pred_lf[1:]
            pred_lf = ''.join(pred_lf).replace('Ġ', ' ').replace('<space>', ' ')
            db_name = ''.join(self.tokenizer.convert_ids_to_tokens(x['db_name'][i])).replace('Ġ', ' ')
            pred_lfs.append((x['id'][i].item(), pred_lf, db_name))
        self.log('val_loss', masked_lm_loss, sync_dist=True, prog_bar=True)
        return {'pred_lfs': pred_lfs, 'loss': masked_lm_loss}

    def validation_step_end(self, step_output):
        pred_dict = {}
        for idx, pred_lf, db_name in step_output['pred_lfs']:
            pred_dict[idx] = (pred_lf, db_name)
        with open(f'bart/predict/predict_rank_{self.global_rank}.txt', 'a') as fa:
            for idx, (pred_lf, db_name) in pred_dict.items():
                fa.write(f'{idx}\t{pred_lf}\t{db_name}\n')
        return pred_dict

    def validation_epoch_end(self, validation_step_output):
        if self.global_rank == 0:
            pred_dict = {}
            for i in range(8):
                if os.path.exists(f'bart/predict/predict_rank_{i}.txt'):
                    with open(f'bart/predict/predict_rank_{i}.txt', 'r') as fr:
                        lines = fr.readlines()
                    for line in lines:
                        idx, pred_lf, db_name = line.strip().split('\t')
                        pred_dict[int(idx)] = (pred_lf, db_name)
                    with open(f'bart/predict/predict_rank_{i}.txt', 'w') as fw:
                        pass
            pred_list = sorted(pred_dict.items(), key=lambda x: x[0])
            with open('bart/predict/predict.txt', 'w') as fw:
                gold = [x.strip() for x in open('sparc/dev_gold.txt', 'r').readlines()]
                for idx, (pred, db_name) in pred_list:
                    fw.write(pred + '\n')
                    del gold[0]
                    if gold[0] == '':
                        fw.write('\n')
                        del gold[0]
            if self.current_epoch % self.check_interval == 0:
                gold = [x.strip() for x in open('sparc/dev_gold.txt', 'r').readlines()]
                with open('bart/predict/predict_debug.txt', 'w') as fw:
                    for i, (pred, db_name) in pred_list:
                        fw.write(f'{i}\t{pred}\t{db_name}\n{i}\t{gold[0]}\n')
                        del gold[0]
                        if gold[0] == '':
                            fw.write('\n')
                            del gold[0]
            exact_match_acc = evaluate_sparc('sparc/dev_gold.txt', 'bart/predict/predict.txt', 'sparc/database', 'sparc/tables.json')
            self.log('val_acc', exact_match_acc, prog_bar=True)

    def test_step(self, x, batch_idx):
        return self.validation_step(x, batch_idx)

    def test_step_end(self, step_output):
        pred_dict = {}
        for idx, pred_lf, db_name in step_output['pred_lfs']:
            pred_dict[idx] = (pred_lf, db_name)
        with open(f'bart/predict/test_rank_{self.global_rank}.txt', 'a') as fa:
            for idx, (pred_lf, db_name) in pred_dict.items():
                fa.write(f'{idx}\t{pred_lf}\t{db_name}\n')
        return pred_dict

    def test_epoch_end(self, test_step_output):
        if self.global_rank == 0:
            pred_dict = {}
            for i in range(8):
                if os.path.exists(f'bart/predict/test_rank_{i}.txt'):
                    with open(f'bart/predict/test_rank_{i}.txt', 'r') as fr:
                        lines = fr.readlines()
                    for line in lines:
                        idx, pred_lf, db_name = line.strip().split('\t')
                        pred_dict[int(idx)] = (pred_lf, db_name)
                    with open(f'bart/predict/test_rank_{i}.txt', 'w') as fw:
                        pass
            pred_list = sorted(pred_dict.items(), key=lambda x: x[0])
            with open('bart/predict/test.txt', 'w') as fw:
                gold = [x.strip() for x in open('sparc/dev_gold.txt', 'r').readlines()]
                for idx, (pred, db_name) in pred_list:
                    fw.write(pred + '\n')
                    del gold[0]
                    if gold[0] == '':
                        fw.write('\n')
                        del gold[0]
            if self.current_epoch % self.check_interval == 0:
                gold = [x.strip() for x in open('sparc/dev_gold.txt', 'r').readlines()]
                with open('bart/predict/test_debug.txt', 'w') as fw:
                    for i, (pred, db_name) in pred_list:
                        fw.write(f'{i}\t{pred}\t{db_name}\n{i}\t{gold[0]}\n')
                        del gold[0]
                        if gold[0] == '':
                            fw.write('\n')
                            del gold[0]
            exact_match_acc = evaluate_sparc('sparc/dev_gold.txt', 'bart/predict/test.txt', 'sparc/database', 'sparc/tables.json')
            self.log('test_acc', exact_match_acc, prog_bar=True)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate)
        return optimizer
