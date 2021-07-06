import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
import time

import torch
from torch import nn
from torch.utils.data import DataLoader, random_split
from torch.nn.utils.rnn import pad_sequence
import torch.distributed as dist
from transformers import BartTokenizer
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

from model import SQLBartModel
from dataset import SparcDataset
from evaluation import evaluate as evaluate_sparc


class SQLBart(pl.LightningModule):
    def __init__(self, tokenizer):
        super().__init__()
        self.tokenizer = tokenizer
        self.model = SQLBartModel()
        self.model.bart_model.resize_token_embeddings(len(self.tokenizer))
        self.loss_fct = torch.nn.CrossEntropyLoss(ignore_index=-100)
        self.learning_rate = 1e-5
        self.check_interval = 1

    def forward(self, x):
        # in lightning, forward defines the prediction/inference actions
        lm_logits = self.model(x)
        pred_ids = lm_logits.argmax(dim=-1)
        pred_str = [self.tokenizer.convert_ids_to_tokens(xx) for xx in pred_ids]
        masked_lm_loss = self.loss_fct(lm_logits.view(-1, self.model.bart_config.vocab_size), x['labels'].view(-1))
        return {'lm_logits': lm_logits, 'sql': pred_str, 'loss': masked_lm_loss}

    def training_step(self, x, batch_idx):
        # training_step defined the train loop.
        # It is independent of forward
        lm_logits = self.model(x)
        masked_lm_loss = self.loss_fct(lm_logits.view(-1, self.model.bart_config.vocab_size), x['labels'].view(-1))

        # Logging to TensorBoard by default
        self.log('train_loss', masked_lm_loss, sync_dist=True)
        return masked_lm_loss

    def validation_step(self, x, batch_idx):
        lm_logits = self.model(x)
        masked_lm_loss = self.loss_fct(lm_logits.view(-1, self.model.bart_config.vocab_size), x['labels'].view(-1))
        pred_ids = lm_logits.argmax(dim=-1)
        pred_lfs = []
        for i in range(pred_ids.size(0)):
            pred_lf = self.tokenizer.convert_ids_to_tokens(pred_ids[i])
            pred_lfs.append((x['id'][i].item(), pred_lf))
        self.log('val_loss', masked_lm_loss, sync_dist=True, prog_bar=True)
        return {'pred_lfs': pred_lfs, 'loss': masked_lm_loss}

    # def validation_step(self, x, batch_idx):
    #     lm_logits = self.model(x)
    #     masked_lm_loss = self.loss_fct(lm_logits.view(-1, self.model.bart_config.vocab_size), x['labels'].view(-1))
    #     pred_ids = lm_logits.argmax(dim=-1)
    #     self.log('val_loss', masked_lm_loss, sync_dist=True, prog_bar=True)
    #     return {'ids': x['id'], 'pred_ids': pred_ids, 'loss': masked_lm_loss}

    def validation_step_end(self, step_output):
        pred_dict = {}
        for idx, pred_lf in step_output['pred_lfs']:
            pred_dict[idx] = pred_lf
        with open(f'bart/predict/predict_rank_{self.global_rank}.txt', 'a') as fa:
            for idx, pred_lf in pred_dict.items():
                fa.write(f'{idx}\t{pred_lf}\n')
        return pred_dict

    def validation_epoch_end(self, validation_step_output):
        if self.global_rank == 0:
            pred_dict = {}
            for i in range(8):
                if os.path.exists(f'bart/predict/predict_rank_{i}.txt'):
                    with open(f'bart/predict/predict_rank_{i}.txt', 'r') as fr:
                        lines = fr.readlines()
                    for line in lines:
                        idx, pred_lf = line.split('\t', 1)
                        pred_dict[idx] = pred_lf
                    with open(f'bart/predict/predict_rank_{i}.txt', 'w') as fw:
                        pass
            pred_list = sorted(pred_dict.items(), key=lambda x: x[0])
            with open('bart/predict/predict.txt', 'w') as fw:
                for idx, pred in pred_list:
                    fw.write(pred)
            if self.current_epoch % self.check_interval == 0:
                gold = [x.strip() for x in open('sparc/dev_gold.txt', 'r').readlines()]
                with open('bart/predict/predict_debug.txt', 'w') as fw:
                    for pred in pred_list:
                        fw.write(f'{pred[1].strip()}\n{gold[0]}\n')
                        del gold[0]
                        if gold[0] == '':
                            fw.write('\n')
                            del gold[0]
            exact_match_acc = evaluate_sparc('sparc/dev_gold.txt', 'bart/predict/predict.txt', 'sparc/database', 'sparc/tables.json')
            self.log('val_acc', exact_match_acc, prog_bar=True)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate)
        return optimizer

    # def optimizer_step(
    #     self, epoch, batch_idx, optimizer, optimizer_idx, optimizer_closure,
    #     on_tpu=False, using_native_amp=False, using_lbfgs=False,
    # ):
    #     if self.trainer.global_step <= 100:
    #         lr_scale = min(1., float(self.trainer.global_step + 1) / 100.)
    #         for pg in optimizer.param_groups:
    #             pg['lr'] = lr_scale * self.learning_rate
    #
    #     optimizer.step(closure=optimizer_closure)


if __name__ == '__main__':
    bart_tokenizer = BartTokenizer.from_pretrained('facebook/bart-large', additional_special_tokens=['<c>', '</c>', '<t>'])
    train_dataset = SparcDataset('sparc/train.json', 'sparc/tables.json', 'sparc/database', tokenizer=bart_tokenizer)
    dev_dataset = SparcDataset('sparc/dev.json', 'sparc/tables.json', 'sparc/database', tokenizer=bart_tokenizer)
    train_dataloader = DataLoader(train_dataset, batch_size=4, shuffle=True, collate_fn=train_dataset.collate_fn)
    dev_dataloader = DataLoader(dev_dataset, batch_size=4, shuffle=False, collate_fn=dev_dataset.collate_fn)

    sql_bart = SQLBart(bart_tokenizer)
    trainer = pl.Trainer(gpus=-1, precision=16, default_root_dir='bart/checkpoints',
                         terminate_on_nan=True, accumulate_grad_batches=2,
                         gradient_clip_val=5, gradient_clip_algorithm='value',
                         callbacks=[EarlyStopping(monitor='val_loss', patience=10, mode='min')])
    trainer.fit(sql_bart, train_dataloader, dev_dataloader)

    trainer.test(test_dataloaders=dev_dataloader)
