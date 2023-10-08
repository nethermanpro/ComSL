import os
import numpy as np
import torchmetrics
import torch
from torch import nn
from torch.utils.data import DataLoader
from pytorch_lightning import LightningModule
from transformers import (
    AdamW,
    get_polynomial_decay_schedule_with_warmup,
)

import sys

if __name__ == "__main__":
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.append(BASE_DIR)

from data.data_util import load_data_record
from data.dataset import MbartDataset
from decode.mbart_decode import decode, DecodingOptions

from model.model_util import load_mbart_tokenizer
from model.mBART_model import MbartModel
from criterion.metric_util import get_segment_tokenizers, preprocess_sentence


class MbartCollatorWhithPadding:
    def __call__(self, features):
        enc_input_ids, labels, dec_input_ids, tgt_lang = [], [], [], []
        for f in features:
            enc_input_ids.append(f["enc_input_ids"])
            labels.append(f["labels"])
            dec_input_ids.append(f["dec_input_ids"])
            tgt_lang.append(f["tgt_lang"])

        label_lengths = [len(lab) for lab in labels]
        dec_input_ids_length = [len(e) for e in dec_input_ids]
        max_label_len = max(label_lengths + dec_input_ids_length)

        labels = [np.pad(lab, (0, max_label_len - lab_len), 'constant', constant_values=pad_token_id) for lab, lab_len
                  in zip(labels, label_lengths)]
        dec_input_ids = [np.pad(e, (0, max_label_len - e_len), 'constant', constant_values=pad_token_id) for e, e_len in
                         zip(dec_input_ids, dec_input_ids_length)]

        enc_input_len = [len(e) for e in enc_input_ids]
        max_enc_input_len = max(enc_input_len)
        enc_input_ids = [np.pad(e, (0, max_enc_input_len - e_len), 'constant', constant_values=pad_token_id) for
                         e, e_len in zip(enc_input_ids, enc_input_len)]

        batch = {
            "labels": labels,
            "dec_input_ids": dec_input_ids,
        }

        batch = {k: torch.tensor(np.array(v), requires_grad=False) for k, v in batch.items()}

        batch["enc_input_ids"] = torch.tensor(np.array(enc_input_ids))
        batch["tgt_lang"] = tgt_lang

        return batch


class MbartModelModule(LightningModule):
    def __init__(self, cfg, joined_dataset: dict, sep_dataset: dict) -> None:
        super().__init__()
        self.tokenizer = load_mbart_tokenizer(cfg)
        global pad_token_id
        pad_token_id = self.tokenizer.pad_token_id
        self.model = MbartModel(cfg)
        self.loss_fn = nn.CrossEntropyLoss(ignore_index=pad_token_id, label_smoothing=cfg.label_smoothing)

        self.cfg = cfg
        self.__train_dataset = joined_dataset.get("train", [])
        self.__eval_dataset = sep_dataset.get("dev", [])
        self.__test_dataset = sep_dataset.get("test", [])

        self.decode_option = DecodingOptions(beam_size=5)
        self.valid_metrics = nn.ModuleDict(
            {"bleu": nn.ModuleList([torchmetrics.BLEUScore(compute_on_step=False) for _ in self.__eval_dataset]),
             "loss": nn.ModuleList([torchmetrics.MeanMetric(compute_on_step=False) for _ in self.__eval_dataset])})

        self.test_metrics = nn.ModuleDict(
            {"bleu": nn.ModuleList([torchmetrics.BLEUScore(compute_on_step=False) for _ in self.__test_dataset])})
        self.segment_tokenizers = get_segment_tokenizers()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_id):
        input_ids = batch["enc_input_ids"]
        bsz = input_ids.size(0)
        labels = batch["labels"].long()
        dec_input_ids = batch["dec_input_ids"].long()

        # with torch.no_grad():
        outputs = self.model(input_ids, dec_input_ids)
        loss = self.loss_fn(outputs.view(-1, outputs.size(-1)), labels.view(-1))
        self.log("train/loss", loss, on_step=True, prog_bar=True, logger=True, on_epoch=True, sync_dist=True,
                 batch_size=bsz)
        return loss

    def on_validation_epoch_start(self) -> None:
        for k, v in self.valid_metrics.items():
            for metric in v:
                metric.set_dtype(torch.float32)

    def validation_step(self, batch, batch_id, dataloader_idx=None):
        input_ids = batch["enc_input_ids"]
        labels = batch["labels"].long()
        dec_input_ids = batch["dec_input_ids"].long()
        tgt_lang = batch['tgt_lang']
        bsz = input_ids.size(0)

        encoder_hidden_states = self.model.encoder(input_ids)
        outputs = self.model.decoder(dec_input_ids, encoder_hidden_states)[0]
        loss = self.loss_fn(outputs.view(-1, outputs.size(-1)), labels.view(-1))
        gens = decode(self.model.decoder,
                      tokenizer=self.tokenizer,
                      enc_hidden_states=encoder_hidden_states,
                      forced_bos_token_id=dec_input_ids[:, 1],
                      options=self.decode_option)

        o_list = self.tokenizer.batch_decode(gens, skip_special_tokens=True)
        l_list = self.tokenizer.batch_decode(labels, skip_special_tokens=True)
        # inputs = self.tokenizer.batch_decode(input_ids, skip_special_tokens=True)

        preprocess_sentence(o_list, tgt_lang, self.segment_tokenizers)
        preprocess_sentence(l_list, tgt_lang, self.segment_tokenizers)

        self.valid_metrics['loss'][dataloader_idx](loss.cuda())
        self.valid_metrics['bleu'][dataloader_idx](o_list, [[l] for l in l_list])

        return {
            "loss": loss,
            "o_list": o_list,
            "l_list": l_list,
        }

    def validation_epoch_end(self, outputs):
        loss_scores = [l.compute() for l in self.valid_metrics['loss']]
        self.log('valid_loss_epoch', torch.mean(torch.tensor(loss_scores)))
        print("valid_loss_epoch", torch.mean(torch.tensor(loss_scores)))
        bleu_scores = [b.compute() * 100 for b in self.valid_metrics['bleu']]
        self.log('valid_bleu_epoch', torch.mean(torch.tensor(bleu_scores)))
        print("valid_bleu_epoch", torch.mean(torch.tensor(bleu_scores)))
        for k, v in self.valid_metrics.items():
            for metric in v:
                metric.reset()

    def on_test_epoch_start(self) -> None:
        for k, v in self.test_metrics.items():
            for metric in v:
                metric.set_dtype(torch.float32)

    def test_step(self, batch, batch_idx, dataloader_idx=None):
        input_ids = batch["enc_input_ids"]
        bsz = input_ids.size(0)
        labels = batch["labels"].long()
        dec_input_ids = batch["dec_input_ids"].long()
        tgt_lang = batch['tgt_lang']
        encoder_hidden_states = self.model.encoder(input_ids)
        gens = decode(self.model.decoder,
                      tokenizer=self.tokenizer,
                      enc_hidden_states=encoder_hidden_states,
                      forced_bos_token_id=dec_input_ids[:, 1],
                      options=self.decode_option)

        o_list = self.tokenizer.batch_decode(gens, skip_special_tokens=True)
        l_list = self.tokenizer.batch_decode(labels, skip_special_tokens=True)

        o_list_ = o_list.copy()
        l_list_ = l_list.copy()

        for i in range(len(o_list)):
            tokenizer = self.segment_tokenizers.get(tgt_lang[i], self.segment_tokenizers["default"])
            o_list[i] = "".join(tokenizer(o_list[i].rstrip()))
            l_list[i] = "".join(tokenizer(l_list[i].rstrip()))
        self.test_metrics['bleu'][dataloader_idx](o_list, [[l] for l in l_list])

        return {
            "output": o_list_,
            "label": l_list_,
        }

    def test_epoch_end(self, outputs):
        bleu_scores = [b.compute() * 100 for b in self.test_metrics['bleu']]
        for i, bleu in enumerate(bleu_scores):
            self.log(f"test_bleu_{i}", round(bleu.item(), 2))
            print(f"test_bleu_{i}", round(bleu.item(), 2))
        self.log('test_bleu_epoch', round(torch.mean(torch.tensor(bleu_scores)).detach().cpu().item(), 2))
        for k, v in self.test_metrics.items():
            for metric in v:
                metric.reset()
        print("test_bleu_epoch", torch.mean(torch.tensor(bleu_scores)))

    def configure_optimizers(self):
        model = self.model
        no_decay = ["bias", "LayerNorm.weight", "layer_norm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in model.named_parameters()
                           if not any(nd in n for nd in no_decay)],
                "weight_decay": self.cfg.weight_decay,
            },
            {
                "params": [p for n, p in model.named_parameters()
                           if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        optimizer = AdamW(optimizer_grouped_parameters,
                          lr=self.cfg.learning_rate,
                          eps=self.cfg.adam_epsilon,
                          betas=self.cfg.adam_betas)
        self.optimizer = optimizer

        scheduler = get_polynomial_decay_schedule_with_warmup(
            optimizer, num_warmup_steps=self.cfg.warmup_steps,
            num_training_steps=self.trainer.estimated_stepping_batches
        )
        self.scheduler = scheduler

        return [optimizer], [{"scheduler": scheduler, "interval": "step", "frequency": 1}]

    def train_dataloader(self):
        dataset = MbartDataset(self.__train_dataset, self.tokenizer)
        return DataLoader(dataset,
                          batch_size=self.cfg.batch_size,
                          drop_last=True, shuffle=True, num_workers=self.cfg.num_worker,
                          collate_fn=MbartCollatorWhithPadding()
                          )

    def val_dataloader(self):
        datasets = [MbartDataset(dataset, self.tokenizer) for dataset in
                    self.__eval_dataset]
        return [DataLoader(dataset,
                           batch_size=self.cfg.test_batch_size,
                           num_workers=self.cfg.num_worker,
                           collate_fn=MbartCollatorWhithPadding()
                           ) for dataset in datasets]

    def test_dataloader(self):
        datasets = [MbartDataset(dataset, self.tokenizer) for dataset in
                    self.__test_dataset]
        return [DataLoader(dataset,
                           batch_size=self.cfg.test_batch_size,
                           num_workers=self.cfg.num_worker,
                           collate_fn=MbartCollatorWhithPadding()
                           ) for dataset in datasets]


if __name__ == "__main__":
    from config.parse_yaml_args import parse_args_and_yaml
    from model.model_util import deep_to_device

    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    cfg = parse_args_and_yaml(config_path="../config/exp_spec/mbart_en_x.yaml")
    DATA_ROOT = cfg.data_root
    cfg.batch_size = cfg.test_batch_size = 10
    language_list = cfg.language_list

    joined_data_pair_lists, sep_data_pair_lists = {}, {}
    for split in ["train", "dev", "test"]:
        joined_data_pair_lists[split], sep_data_pair_lists[split] = load_data_record(DATA_ROOT, split,
                                                                                     language_list=language_list)
    module = MbartModelModule(cfg, joined_data_pair_lists, sep_data_pair_lists).cuda().eval()

    loader = module.test_dataloader()[0]

    with torch.no_grad():
        for b in loader:
            b = deep_to_device(b, cfg.device)
            train_res = module.training_step(b, 0)
            print(train_res)

            valid_res = module.validation_step(b, 0, 0)
            print(valid_res)
            module.validation_epoch_end([valid_res])

            test_res = module.test_step(b, 0, 0)
            print(test_res)
            module.test_epoch_end([test_res])

            break
