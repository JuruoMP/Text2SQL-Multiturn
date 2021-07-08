import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

from .model import SQLBart
from .dataset import SparcDataset
from .tokenization_bart import BartTokenizer


if __name__ == '__main__':
    bart_tokenizer = BartTokenizer.from_pretrained('facebook/bart-large', additional_special_tokens=['<c>'])
    train_dataset = SparcDataset('sparc/train.json', 'sparc/tables.json', 'sparc/database', tokenizer=bart_tokenizer)
    dev_dataset = SparcDataset('sparc/dev.json', 'sparc/tables.json', 'sparc/database', tokenizer=bart_tokenizer)
    train_dataloader = DataLoader(train_dataset, batch_size=4, shuffle=True, collate_fn=train_dataset.collate_fn)
    dev_dataloader = DataLoader(dev_dataset, batch_size=4, shuffle=False, collate_fn=dev_dataset.collate_fn)

    sql_bart = SQLBart(bart_tokenizer)
    trainer = pl.Trainer(gpus=-1, precision=16, default_root_dir='bart/checkpoints',
                         terminate_on_nan=True, accumulate_grad_batches=1,
                         gradient_clip_val=5, gradient_clip_algorithm='value',
                         callbacks=[EarlyStopping(monitor='val_loss', patience=10, mode='min')])
    trainer.fit(sql_bart, train_dataloader, dev_dataloader)

    trainer.test(test_dataloaders=dev_dataloader)
