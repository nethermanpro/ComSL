from torch import nn
from model.model_util import load_mbart_model


class MbartDecoder(nn.Module):
    def __init__(self, cfg, mbart_model) -> None:
        super().__init__()
        self.cfg = cfg
        self.decoder = mbart_model.base_model.decoder
        self.lm_head = mbart_model.lm_head

    def forward(self, dec_input_ids, encoder_hidden_states, past_key_values=None, use_cache=False):
        dec_output = self.decoder(dec_input_ids,
                                  encoder_hidden_states=encoder_hidden_states,
                                  past_key_values=past_key_values,
                                  use_cache=use_cache)
        last_hidden_state = dec_output.last_hidden_state
        if use_cache:
            attn_key_values = dec_output.past_key_values
        lm_logits = self.lm_head(last_hidden_state)
        if use_cache:
            return [lm_logits, {'attn_key_values': attn_key_values}]
        else:
            return [lm_logits, {}]


class MbartEncoder(nn.Module):
    def __init__(self, cfg, mbart_model) -> None:
        super().__init__()
        self.cfg = cfg

        self.encoder = mbart_model.base_model.encoder

    def forward(self, input_ids, attention_mask=None, output_attentions=False):
        return self.encoder(input_ids, attention_mask=attention_mask,
                            output_attentions=output_attentions).last_hidden_state


class MbartModel(nn.Module):
    def __init__(self, cfg) -> None:
        super().__init__()
        mbart_model = load_mbart_model(cfg)
        self.encoder = MbartEncoder(cfg, mbart_model)
        self.decoder = MbartDecoder(cfg, mbart_model)

    def forward(self, enc_input_ids, dec_input_ids, attention_mask=None):
        encoder_hidden_states = self.encoder(enc_input_ids, attention_mask=attention_mask)
        lm_logits = self.decoder(dec_input_ids, encoder_hidden_states)[0]
        return lm_logits