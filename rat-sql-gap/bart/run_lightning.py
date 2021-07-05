import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

import torch
from torch import nn
from torch.utils.data import DataLoader, random_split
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
        self.log('train_loss', masked_lm_loss)
        return masked_lm_loss

    def validation_step(self, x, batch_idx):
        lm_logits = self.model(x)
        masked_lm_loss = self.loss_fct(lm_logits.view(-1, self.model.bart_config.vocab_size), x['labels'].view(-1))
        self.log('val_loss', masked_lm_loss, prog_bar=True)
        pred_ids = lm_logits.argmax(dim=-1)
        pred_lfs = []
        for i in range(pred_ids.size(0)):
            pred_lf = self.tokenizer.convert_ids_to_tokens(pred_ids[i])
            pred_lfs.append((x['id'][i].item(), pred_lf))
        return (pred_lfs, masked_lm_loss)

    def validation_epoch_end(self, val_ret):
        if self.global_rank == 0:
            pred_list = [j for i in val_ret for j in i[0]]
            losses = [i[1].item() for i in val_ret]
            avg_loss = sum(losses) / len(losses)
            new_pred_list = []
            for i in range(len(pred_list)):
                idx, pred = pred_list[i]
                if self.tokenizer.eos_token in pred:
                    pred = pred[1:pred.index(self.tokenizer.eos_token)]
                else:
                    pred = pred[1:]
                pred = ''.join(pred).replace('Ġ', ' ')
                new_pred_list.append((idx, pred))

            gold = open('sparc/dev_gold.txt', 'r', encoding='utf-8').readlines()
            if not os.path.exists('bart/predict'):
                os.makedirs('bart/predict')
            new_pred_list = sorted(new_pred_list, key=lambda x: x[0])
            with open('bart/predict/predict.txt', 'w') as fw:
                for idx, pred in new_pred_list:
                    fw.write(pred + '\n')
                    del gold[0]
                    if gold[0].strip() == '':
                        del gold[0]
                        fw.write('\n')
            if self.current_epoch % 10 == 0:
                with open(f'bart/predict/predict_{self.current_epoch}.txt', 'w') as fw:
                    with open('bart/predict/predict.txt', 'r') as fr:
                        fw.writelines(fr.readlines())
            exact_match_acc = evaluate_sparc('sparc/dev_gold.txt', 'bart/predict/predict.txt', 'sparc/database', 'sparc/tables.json')
            self.log('val_acc', exact_match_acc, prog_bar=True)
            # print(f'Validation exact match acc = {exact_match_acc:.3f}, loss = {avg_loss:.3e}')

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate)
        return optimizer

    def optimizer_step(
        self, epoch, batch_idx, optimizer, optimizer_idx, optimizer_closure,
        on_tpu=False, using_native_amp=False, using_lbfgs=False,
    ):
        if self.trainer.global_step <= 100:
            lr_scale = min(1., float(self.trainer.global_step + 1) / 100.)
            for pg in optimizer.param_groups:
                pg['lr'] = lr_scale * self.learning_rate

        optimizer.step(closure=optimizer_closure)


if __name__ == '__main__':
    bart_tokenizer = BartTokenizer.from_pretrained('facebook/bart-large', additional_special_tokens=['<c>', '</c>', '<t>', '</t>'])
    train_dataset = SparcDataset('sparc/train.json', 'sparc/tables.json', 'sparc/database', tokenizer=bart_tokenizer)
    dev_dataset = SparcDataset('sparc/dev.json', 'sparc/tables.json', 'sparc/database', tokenizer=bart_tokenizer)
    train_dataloader = DataLoader(train_dataset, batch_size=8, shuffle=True, collate_fn=train_dataset.collate_fn)
    dev_dataloader = DataLoader(dev_dataset, batch_size=8, shuffle=False, collate_fn=dev_dataset.collate_fn)

    sql_bart = SQLBart(bart_tokenizer)
    trainer = pl.Trainer(gpus=-1, precision=16, default_root_dir='bart/checkpoints',
                         terminate_on_nan=True,
                         gradient_clip_val=5, gradient_clip_algorithm='value',
                         callbacks=[EarlyStopping(monitor='val_loss', patience=10, mode='min')])
    trainer.fit(sql_bart, train_dataloader, dev_dataloader)

    trainer.test(test_dataloaders=dev_dataloader)
