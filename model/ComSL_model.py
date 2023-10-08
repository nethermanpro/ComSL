import os
import torch
import torch.nn as nn
from torch import Tensor
from model.model_util import Conv1dAdaptor, GradMultiply, load_mbart_model
import Whisper


class ComSTDecoder(nn.Module):
    def __init__(self, cfg, mbart_model) -> None:
        super().__init__()

        self.decoder = TextDecoder(cfg, mbart_model)
        self.d_model = 1024
        self.cfg = cfg
        self.num_updates = 0

    def forward(self, dec_input_ids, encoder_out, mlm_mask=None, past_key_values=None, use_cache=False):

        if isinstance(encoder_out, tuple):  # only for training
            assert len(encoder_out) == 2 and past_key_values is None and use_cache is False

            transcript_dec_inputs, translate_dec_inputs = dec_input_ids

            dec_outs = {'translate': [], 'transcript': []}
            for i in range(2):
                if transcript_dec_inputs is not None:
                    if i == 0 and self.cfg.asr_loss_weight > 0:
                        logits, misc_dict = self.decoder(transcript_dec_inputs, encoder_out[i]['encoder_out'])
                    else:
                        logits = None
                    dec_outs['transcript'].append(logits)

                if translate_dec_inputs is not None:
                    if i == 0 and self.cfg.spch_loss_weight == 0:
                        logits = None
                    elif i == 1 and self.cfg.text_loss_weight == 0:
                        logits = None
                    else:
                        logits, misc_dict = self.decoder(translate_dec_inputs, encoder_out[i]['encoder_out'])
                    dec_outs['translate'].append(logits)

            mix_dec_outs = None
            erm_loss = None
            if self.cfg.use_cml and 'mix_enc_out_spch' in encoder_out[1]:
                mix_enc_out = [encoder_out[1]['mix_enc_out_spch'], encoder_out[1]['mix_enc_out_text']]
                mix_dec_outs = {'translate': [], 'transcript': []}
                for i in range(2):
                    if transcript_dec_inputs is not None:
                        logits, misc_dict = self.decoder(transcript_dec_inputs, mix_enc_out[i])
                        mix_dec_outs['transcript'].append(logits)

                    if translate_dec_inputs is not None:
                        logits, misc_dict = self.decoder(translate_dec_inputs, mix_enc_out[i])
                        mix_dec_outs['translate'].append(logits)

                if self.cfg.use_erm:
                    erm_loss = (encoder_out[0]['reg_hidden'] - encoder_out[1]['reg_hidden']).norm(2, dim=-1)

            return dec_outs, {"erm_loss": erm_loss, "mix_dec_outs": mix_dec_outs, 'mlm_mask': mlm_mask}

        else:  # for decoding
            if isinstance(encoder_out, Tensor):
                encoder_out = {"encoder_out": encoder_out, "encoder_padding_mask": None}
            return self.decoder(dec_input_ids, encoder_out['encoder_out'],
                                encoder_padding_mask=encoder_out["encoder_padding_mask"],
                                past_key_values=past_key_values,
                                use_cache=use_cache)


class TextDecoder(nn.Module):
    def __init__(self, cfg, mbart_model) -> None:
        super().__init__()
        self.decoder = mbart_model.base_model.decoder
        self.lm_head = mbart_model.lm_head
        self.embed_tokens = self.decoder.embed_tokens
        self.embed_scale = self.decoder.embed_scale
        self.cfg = cfg

    def forward(self, dec_input_ids, encoder_out, encoder_padding_mask=None, past_key_values=None, use_cache=False):
        output_attentions = True if past_key_values is None else False
        text_embeds = self.embed_tokens(dec_input_ids) * self.embed_scale

        dec_output = self.decoder(inputs_embeds=text_embeds,
                                  encoder_hidden_states=encoder_out,
                                  encoder_attention_mask=encoder_padding_mask,
                                  past_key_values=past_key_values,
                                  use_cache=use_cache,
                                  output_attentions=output_attentions)
        last_hidden_state = dec_output.last_hidden_state

        attn_key_values = dec_output.past_key_values if use_cache else None
        logits = self.lm_head(last_hidden_state)
        return logits, {"attn_key_values": attn_key_values}


class TextEncoder(nn.Module):
    def __init__(self, cfg, text_encoder) -> None:
        super().__init__()
        self.encoder = text_encoder
        self.cfg = cfg
        self.embed_tokens = text_encoder.embed_tokens
        self.embed_scale = text_encoder.embed_scale
        self.erm_layer = cfg.erm_layer

    def forward(self, src_tokens, masked_src_tokens, spch_embeds, **kwargs):
        enc_out = {}
        inputs_embeds = self.embed_tokens(src_tokens) * self.embed_scale
        enc_out['embeds'] = inputs_embeds
        encoder_out = self.encoder(inputs_embeds=inputs_embeds, attention_mask=None, output_hidden_states=True)
        enc_out['encoder_out'] = encoder_out.last_hidden_state
        enc_out['encoder_padding_mask'] = None
        text_hidden_ori = encoder_out.hidden_states[self.erm_layer]
        enc_out['text_hidden_ori'] = text_hidden_ori / text_hidden_ori.norm(2, dim=-1, keepdim=True)

        if self.cfg.use_cml and masked_src_tokens is not None:
            masked_inputs_embeds = self.embed_tokens(masked_src_tokens) * self.embed_scale
            text_len = masked_inputs_embeds.shape[1]
            mix_input_embeds = torch.cat([masked_inputs_embeds, spch_embeds], dim=1)
            mix_enc_out = self.encoder(inputs_embeds=mix_input_embeds, attention_mask=None, output_hidden_states=True)
            enc_out['mix_enc_out_text'] = mix_enc_out.last_hidden_state[:, :text_len, :]
            enc_out['mix_enc_out_spch'] = mix_enc_out.last_hidden_state[:, text_len:, :]
            text_hidden_mix = mix_enc_out.hidden_states[self.erm_layer][:, :text_len, :]
            reg_hidden = mix_enc_out.hidden_states[self.erm_layer][:, text_len:, :]
            enc_out['reg_hidden'] = reg_hidden / reg_hidden.norm(2, dim=-1, keepdim=True)
            enc_out['text_hidden_mix'] = text_hidden_mix / text_hidden_mix.norm(2, dim=-1, keepdim=True)

        return enc_out


class SpchEncoder(nn.Module):

    def __init__(self, cfg, text_encoder) -> None:
        super().__init__()
        whisper_model = Whisper.load_model(
            cfg.whisper_name,
            download_root=cfg.cache_dir,
            device='cpu'
        )
        if cfg.spch_init_model_path is not None:
            whisper_model.load_state_dict(torch.load(os.path.join(cfg.cache_dir, cfg.spch_init_model_path),
                                                     map_location="cpu"))
            print("loaded asr model from {}".format(os.path.join(cfg.cache_dir, cfg.spch_init_model_path)))
        self.spch_encoder = whisper_model.encoder
        self.spch_encoder.gradient_checkpointing = cfg.use_acti_ckpt
        if cfg.spch_n_layers > 0:
            self.spch_encoder.blocks = self.spch_encoder.blocks[:cfg.spch_n_layers]
        self.down_sampler1 = Conv1dAdaptor(whisper_model.dims.n_audio_state, 1024, n_layers=1, proj=True)
        self.down_sampler2 = Conv1dAdaptor(1024, 1024, n_layers=1, )
        self.mbart_encoder = text_encoder
        self.erm_layer = cfg.erm_layer
        self.cfg = cfg

        self.embed_tokens = text_encoder.embed_tokens
        self.embed_scale = text_encoder.embed_scale

    def forward(self, mel, src_lang_ids):
        enc_out = {}
        audio_embeds = self.down_sampler1(self.spch_encoder(mel))[0]
        inputs_embeds = self.down_sampler2(audio_embeds)[0]
        lang_embeds = self.embed_tokens(src_lang_ids.reshape(-1, 1)) * self.embed_scale
        inputs_embeds = torch.cat([lang_embeds, inputs_embeds], dim=1)
        encoder_out = self.mbart_encoder(inputs_embeds=inputs_embeds, attention_mask=None, output_hidden_states=True)
        regulation_hidden = encoder_out.hidden_states[self.erm_layer]

        enc_out['encoder_out'] = encoder_out.last_hidden_state
        enc_out['encoder_padding_mask'] = None
        enc_out['reg_hidden'] = regulation_hidden / regulation_hidden.norm(dim=-1, keepdim=True)
        enc_out['embeds'] = inputs_embeds
        return enc_out


class ComSTEncoder(nn.Module):
    def __init__(
            self,
            args,
            mbart_model,
    ):
        super().__init__()

        mbart_encoder = mbart_model.base_model.encoder

        self.spch_encoder = SpchEncoder(args, mbart_encoder)
        self.text_encoder = TextEncoder(args, mbart_encoder)
        self.enc_grad_mult = args.enc_grad_mult

    def mult_rst_grad(self, rst, ratio):
        assert isinstance(rst, dict)  # instead of EncoderOut
        rst["encoder_out"] = GradMultiply.apply(rst["encoder_out"], ratio)
        return rst

    def forward(
            self,
            mel,
            src_tokens,
            src_lang_ids,
            masked_src_tokens,
            **kwargs
    ):

        if mel is None and src_tokens is None:
            raise ValueError(
                "src_tokens and src_txt_tokens cannot be None at the same time"
            )
        ret1 = None
        ret2 = None
        if mel is not None:
            ret1 = self.spch_encoder(mel, src_lang_ids)

        if src_tokens is not None:
            ret2 = self.text_encoder(
                src_tokens,
                masked_src_tokens,
                ret1['embeds'],
                **kwargs
            )

        def merge_output(rst1, rst2):
            if self.enc_grad_mult != 1.0 and self.training:
                rst1 = self.mult_rst_grad(rst1, self.enc_grad_mult)
                rst2 = self.mult_rst_grad(rst2, self.enc_grad_mult)
            rst = (rst1, rst2)
            return rst

        return merge_output(ret1, ret2)


class ComSTModel(nn.Module):
    def __init__(self, args):
        super().__init__()
        mbart_model = load_mbart_model(args, load_from_local=True, path=args.language_init_model_path)
        self.encoder = ComSTEncoder(args, mbart_model)
        self.decoder = ComSTDecoder(args, mbart_model)

        self.args = args
        self.num_updates = 0
        if args.language_regularization_model_path is not None:
            self.reg_model = load_mbart_model(args, load_from_local=True,
                                              path=args.language_regularization_model_path).eval()
            self.reg_model.requires_grad_(False)
        else:
            self.reg_model = None

    def set_num_updates(self, num_updates, current_epoch=None):
        """Set the number of parameters updates."""
        self.num_updates = num_updates
        self.decoder.num_updates = num_updates
        if self.args.disable_spch_grad_epoch > 0:
            if current_epoch < self.args.disable_spch_grad_epoch:
                self.encoder.spch_encoder.spch_encoder.requires_grad_(False)
            else:
                self.encoder.spch_encoder.spch_encoder.requires_grad_(True)

    def forward(
            self,
            mel,
            src_lang_ids,
            tokens,
            masked_src_tokens,
            mlm_mask,
            **kwargs
    ):
        encoder_out = self.encoder(
            mel,
            src_tokens=tokens[0],
            src_lang_ids=src_lang_ids,
            masked_src_tokens=masked_src_tokens,
            **kwargs
        )
        decoder_out = self.decoder(
            tokens,
            encoder_out=encoder_out,
            mlm_mask=mlm_mask,
        )
        if self.reg_model is not None:
            with torch.no_grad():
                reg_logits = self.reg_model(
                    input_ids=tokens[0],
                    decoder_input_ids=tokens[1],
                ).logits
            decoder_out[1]['reg_logits'] = reg_logits

        return decoder_out, encoder_out
