import torch
import torch.nn as nn
from transformers import BartModel, BartTokenizer, BartConfig


class SQLBartModel(nn.Module):
    def __init__(self):
        super().__init__()
        config_name = 'facebook/bart-large'
        self.bart_config = BartConfig.from_pretrained(config_name, cache_dir='bart/cache')
        self.bart_tokenizer = BartTokenizer.from_pretrained(config_name, cache_dir='bart/cache')
        self.bart_model = BartModel.from_pretrained(config_name, cache_dir='bart/cache')
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
