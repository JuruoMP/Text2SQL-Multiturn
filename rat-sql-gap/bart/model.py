import os
import torch
import pytorch_lightning as pl

from configuration_bart import BartConfig
from modeling_bart import BartPretrainedModel, BartModel, BartForConditionalGeneration, shift_tokens_right
from evaluation import evaluate as evaluate_sparc


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

    @staticmethod
    def shift_tokens_right(input_ids: torch.Tensor, pad_token_id: int, decoder_start_token_id: int):
        shifted_input_ids = input_ids.new_zeros(input_ids.shape)
        shifted_input_ids[:, 1:] = input_ids[:, :-1].clone()
        shifted_input_ids[:, 0] = decoder_start_token_id
        assert pad_token_id is not None, "self.model.config.pad_token_id has to be defined."
        # replace possible -100 values in labels by `pad_token_id`
        shifted_input_ids.masked_fill_(shifted_input_ids == -100, pad_token_id)
        return shifted_input_ids

    def training_step(self, x, batch_idx):
        masked_lm_loss = self.model(
            input_ids=x['input_ids'],
            attention_mask=x['attention_mask'],
            decoder_input_ids=self.shift_tokens_right(x['decoder_input_ids'], self.tokenizer.pad_token_id, self.tokenizer.eos_token_id),
            decoder_attention_mask=x['decoder_attention_mask'],
            labels=x['labels']
        ).loss
        self.log('train_loss', masked_lm_loss, sync_dist=True)
        return masked_lm_loss

    def validation_step(self, x, batch_idx):
        masked_lm_loss = self.model(
            input_ids=x['input_ids'],
            attention_mask=x['attention_mask'],
            decoder_input_ids=self.shift_tokens_right(x['decoder_input_ids'], self.tokenizer.pad_token_id, self.tokenizer.eos_token_id),
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
