import torch
import torch.nn as nn
from transformers import BartModel, BartTokenizer, BartConfig


class SQLBart(nn.Module):
    def __init__(self):
        super().__init__()
        config_name = 'facebook/bart-large'
        self.bart_config = BartConfig.from_pretrained(config_name)
        self.bart_model = BartModel.from_pretrained(config_name)
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

        loss_fct = torch.nn.CrossEntropyLoss(ignore_index=-100)
        masked_lm_loss = loss_fct(lm_logits.view(-1, self.bart_config.vocab_size), x['labels'].view(-1))

        return {'logits': lm_logits, 'loss': masked_lm_loss}
