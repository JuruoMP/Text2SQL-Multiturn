import os
import torch
import torch.nn as nn
import pytorch_lightning as pl

from configuration_bart import BartConfig
from modeling_bart import BartModel
from evaluation import evaluate as evaluate_sparc


class SQLBartModel(nn.Module):
    def __init__(self, name_or_path):
        super().__init__()
        self.bart_config = BartConfig.from_pretrained(name_or_path, cache_dir='bart/cache')
        self.bart_model = BartModel.from_pretrained(name_or_path, cache_dir='bart/cache')
        self.register_buffer("final_logits_bias", torch.zeros((1, self.bart_model.shared.num_embeddings)))
        self.lm_head = nn.Linear(self.bart_config.d_model, self.bart_model.shared.num_embeddings, bias=False)

    def forward(self, x):
        bart_outputs = self.bart_model(
            input_ids=x['input_ids'],
            attention_mask=x['attention_mask'],
            decoder_input_ids=x['decoder_input_ids'],
            decoder_attention_mask=x['decoder_attention_mask']
        )
        lm_logits = self.lm_head(bart_outputs[0]) + self.final_logits_bias

        return lm_logits


class SQLBart(pl.LightningModule):
    def __init__(self, tokenizer):
        super().__init__()
        self.tokenizer = tokenizer
        self.model = SQLBartModel(tokenizer.name_or_path)
        self.model.bart_model.resize_token_embeddings(len(self.tokenizer))
        self.loss_fct = torch.nn.CrossEntropyLoss(ignore_index=-100)
        self.learning_rate = 1e-5
        self.check_interval = 1

    def forward(self, x):
        lm_logits = self.model(x)
        pred_ids = lm_logits.argmax(dim=-1)
        pred_str = [self.tokenizer.convert_ids_to_tokens(xx) for xx in pred_ids]
        masked_lm_loss = self.loss_fct(lm_logits.view(-1, self.model.bart_config.vocab_size), x['labels'].view(-1))
        return {'lm_logits': lm_logits, 'sql': pred_str, 'loss': masked_lm_loss}

    def training_step(self, x, batch_idx):
        lm_logits = self.model(x)
        masked_lm_loss = self.loss_fct(lm_logits.view(-1, self.model.bart_config.vocab_size), x['labels'].view(-1))
        self.log('train_loss', masked_lm_loss, sync_dist=True)
        return masked_lm_loss

    def validation_step(self, x, batch_idx):
        lm_logits = self.model(x)
        masked_lm_loss = self.loss_fct(lm_logits.view(-1, self.model.bart_config.vocab_size), x['labels'].view(-1))
        pred_ids = lm_logits.argmax(dim=-1)
        pred_lfs = []
        for i in range(pred_ids.size(0)):
            pred_lf = self.tokenizer.convert_ids_to_tokens(pred_ids[i])
            if self.tokenizer.eos_token in pred_lf:
                pred_lf = pred_lf[1:pred_lf.index(self.tokenizer.eos_token)]
            else:
                pred_lf = pred_lf[1:]
            pred_lf = ''.join(pred_lf).replace('Ġ', ' ')
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

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate)
        return optimizer
