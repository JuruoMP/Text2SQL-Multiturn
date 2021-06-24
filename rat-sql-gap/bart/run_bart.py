import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

import torch
from tqdm import tqdm

from dataset import SparcDataset, DataLoader
from model import SQLBartModel


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    train_dataset = SparcDataset('sparc/train.json', 'sparc/tables.json', 'sparc/database')
    dev_dataset = SparcDataset('sparc/dev.json', 'sparc/tables.json', 'sparc/database')
    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=train_dataset.collate_fn)
    dev_dataloader = DataLoader(dev_dataset, batch_size=32, shuffle=False, collate_fn=dev_dataset.collate_fn)
    sql_bart = SQLBartModel().to(device)
    if torch.cuda.device_count() > 1:
        sql_bart = torch.nn.DataParallel(sql_bart)

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
            if accu_step % 100 == 0:
                print(f'Avg loss = {accu_loss / accu_step:.2e}')
                accu_loss, accu_step = 0.0, 0

        if epoch % 12 == 0:
            sql_bart.eval()
            all_pred_s = []
            for batch in dev_dataloader:
                pred_logits = sql_bart(batch)['logits']
                pred_ids = pred_logits.argmax(dim=-1)
                for s in pred_ids:
                    pred_s = dev_dataset.tokenizer.convert_ids_to_tokens(s)
                    all_pred_s.append(' '.join(pred_s))
            if not os.path.exists('bart/log'):
                os.makedirs('bart/log')
            with open(f'bart/log/{epoch}.txt', 'w', encoding='utf-8') as fw:
                for s in all_pred_s:
                    fw.write(s + '\n')


if __name__ == '__main__':
    main()
