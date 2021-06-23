import os
import json
import torch
from transformers import BartModel, BartTokenizer, BartConfig
from tqdm import tqdm

from dataset import SparcDataset, DataLoader
from model import SQLBart


def main():
    device = 'cpu' # 'cuda' if torch.cuda.is_available() else 'cpu'
    train_dataset = SparcDataset('sparc/train.json', 'sparc/tables.json', 'sparc/database')
    dev_dataset = SparcDataset('sparc/dev.json', 'sparc/tables.json', 'sparc/database')
    train_dataloader = DataLoader(train_dataset, batch_size=6, shuffle=True, collate_fn=train_dataset.collate_fn)
    dev_dataloader = DataLoader(dev_dataset, batch_size=6, shuffle=False, collate_fn=dev_dataset.collate_fn)
    sql_bart = SQLBart().to(device)

    optimizer = torch.optim.AdamW(sql_bart.parameters())

    for epoch in range(1000):
        sql_bart.train()
        accu_loss, accu_step = 0.0, 0
        for batch in tqdm(train_dataloader):
            with torch.autograd.set_detect_anomaly(True):
                loss = sql_bart(batch)['loss']
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            accu_loss += loss.item()
            accu_step += 1
            if accu_step % 10 == 0:
                print(f'Avg loss = {accu_loss / accu_step:.2e}')
                accu_loss, accu_step = 0.0, 0

        # if epoch % 10 == 0:
        #     sql_bart.eval()
        #     for batch in dev_dataloader:
        #         pred_logits = sql_bart(batch)['logits']
        #         pred_ids = pred_logits.argmax(dim=-1)


if __name__ == '__main__':
    main()
