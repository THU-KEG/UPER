# coding: utf-8

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BartTokenizer, BartForConditionalGeneration, BartConfig
from transformers import LEDForConditionalGeneration, LEDTokenizer


class BartDecodeModel(nn.Module):
    def __init__(self, model_name='facebook/bart-large'):
        super(BartDecodeModel, self).__init__()
        # self.bart_tokenizer = BartTokenizer.from_pretrained('/data1/tsq/WikiGen/pretrained_models/bart_base')
        self.model_name = model_name
        if 'bart' in model_name:
            try:
                self.bart_tokenizer = BartTokenizer.from_pretrained(model_name)
                self.bart_model = BartForConditionalGeneration.from_pretrained(model_name)
            except ValueError:
                self.bart_tokenizer = BartTokenizer.from_pretrained(
                    '/data1/tsq/WikiGen/pretrained_models/bart_base')
                self.bart_model = BartForConditionalGeneration.from_pretrained(
                    '/data1/tsq/WikiGen/pretrained_models/bart_base')
        elif model_name == "allenai/led-base-16384":
            try:
                self.bart_tokenizer = LEDTokenizer.from_pretrained(model_name)
                self.bart_model = LEDForConditionalGeneration.from_pretrained(model_name, gradient_checkpointing=True,
                                                                              use_cache=False)
            except ValueError:
                self.bart_tokenizer = LEDTokenizer.from_pretrained(
                    '/data1/tsq/WikiGen/pretrained_models/led')
                self.bart_model = LEDForConditionalGeneration.from_pretrained(
                    '/data1/tsq/WikiGen/pretrained_models/led')

    def bart_summarize(self, input_ids, attention_mask, global_attention_mask, num_beams=16, max_length=140,
                       min_length=55, device='cuda'):

        if "led" in self.model_name:
            summary_ids = self.bart_model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                global_attention_mask=global_attention_mask,
                num_return_sequences=num_beams, num_beam_groups=num_beams, diversity_penalty=1.0, num_beams=num_beams,
                max_length=max_length + 2,  # +2 from original because we start at step=1 and stop before max_length
                min_length=min_length + 1,  # +1 from original because we start at step=1
                no_repeat_ngram_size=3,
                length_penalty=2.0,
                early_stopping=True)
        else:
            summary_ids = self.bart_model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                num_return_sequences=num_beams, num_beam_groups=num_beams, diversity_penalty=1.0, num_beams=num_beams,
                max_length=max_length + 2,  # +2 from original because we start at step=1 and stop before max_length
                min_length=min_length + 1,  # +1 from original because we start at step=1
                no_repeat_ngram_size=3,
                length_penalty=2.0,
                early_stopping=True)
        # list of str
        abstract_list = [self.bart_tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=False) for
                         g in
                         summary_ids]
        return abstract_list

    def forward(self, input_ids, attention_mask, labels=None, num_beams=16, max_length=140, min_length=55,
                device='cuda'):
        global_attention_mask = torch.zeros_like(input_ids)
        global_attention_mask[:, 0] = 1

        if labels != None:
            # is training
            if "led" in self.model_name:
                loss = self.bart_model(input_ids, attention_mask=attention_mask,
                                       global_attention_mask=global_attention_mask,
                                       labels=labels)[0]
            else:
                loss = self.bart_model(input_ids, attention_mask=attention_mask, labels=labels)[0]
            return loss
        else:
            sum_str_list = self.bart_summarize(input_ids, attention_mask, global_attention_mask, num_beams, max_length,
                                               min_length,
                                               device='cuda')
            return sum_str_list
