import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

from model import SQLBart
from dataset import SparcDataModule
from tokenization_bart import BartTokenizer


if __name__ == '__main__':
    config_name = 'facebook/bart-base'
    bart_tokenizer = BartTokenizer.from_pretrained(config_name, additional_special_tokens=['<c>', '<space>'])
    sparc_data = SparcDataModule('sparc/', batch_size=8, tokenizer=bart_tokenizer)

    sql_bart = SQLBart(bart_tokenizer)
    trainer = pl.Trainer(gpus=-1, precision=16, default_root_dir='bart/checkpoints',
                         terminate_on_nan=True, accumulate_grad_batches=1,
                         gradient_clip_val=5, gradient_clip_algorithm='value',
                         callbacks=[EarlyStopping(monitor='val_loss', patience=10, mode='min')],
                         resume_from_checkpoint='bart/checkpoints/lightning_logs/version_0/checkpoints/epoch=30-step=8648.ckpt'
                         )
    trainer.fit(model=sql_bart, datamodule=sparc_data)

    trainer.test(model=sql_bart, datamodule=sparc_data)
