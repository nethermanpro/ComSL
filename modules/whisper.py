import os
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader
from pytorch_lightning import LightningModule
import torchmetrics
from transformers import WhisperTokenizer

if __name__ == "__main__":
    import sys

    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.append(BASE_DIR)

from decode.whisper_decode import decode, DecodingOptions
from data.data_util import LANG_DICT, load_data_record, pad_trim_audio
from data.dataset import CoVoSTDataset
from model.optimizer import configure_optimizer_schedular
from criterion.metric_util import get_segment_tokenizers, preprocess_sentence
import Whisper


class WhisperTranslateDataset(CoVoSTDataset):
    def __init__(self, audio_info_list, tokenizer, sample_rate=16000) -> None:
        super().__init__(audio_info_list, tokenizer, sample_rate)

    def tokenize(self, src_lang, tgt_lang, transcription, translation):
        self.tokenizer.set_prefix_tokens(language=src_lang)
        translation_ids = self.tokenizer(text=translation, return_tensors="np", max_length=self.max_text_length,
                                         truncation=True).input_ids[0].tolist()
        translation_labels = translation_ids[1:] + [self.tokenizer.eos_token_id]
        return None, translation_ids, None, translation_labels


class WhisperDataCollatorWhithPadding:
    def __init__(self, cfg, pad_token_id=-100):
        self.cfg = cfg
        self.pad_token_id = pad_token_id

    def __call__(self, features):
        audio, labels, dec_input_ids, audio_paths, src_lang, tgt_lang = [], [], [], [], [], []
        for f in features:
            audio.append(f["audio"])
            labels.append(f["translation_labels"])
            dec_input_ids.append(f["translation_ids"])
            audio_paths.append(f["audio_path"])
            src_lang.append(LANG_DICT[f["src_lang"]]['whisper'])
            tgt_lang.append(LANG_DICT[f["tgt_lang"]]['whisper'])

        # audio
        audio_input_feature = pad_trim_audio(audio, self.cfg)

        label_lengths = [len(lab) for lab in labels]
        dec_input_ids_length = [len(e) for e in dec_input_ids]
        max_label_len = max(label_lengths + dec_input_ids_length)

        labels = [np.pad(lab, (0, max_label_len - lab_len), 'constant', constant_values=-100) for lab, lab_len in
                  zip(labels, label_lengths)]
        dec_input_ids = [np.pad(e, (0, max_label_len - e_len), 'constant', constant_values=50257) for e, e_len in
                         zip(dec_input_ids, dec_input_ids_length)]  # 50257 is eot token id

        batch = {
            "labels": labels,
            "dec_input_ids": dec_input_ids,
        }

        batch = {k: torch.tensor(np.array(v), requires_grad=False) for k, v in batch.items()}
        batch["input_ids"] = audio_input_feature
        batch["audio_paths"] = audio_paths
        batch["src_lang"] = src_lang
        batch["tgt_lang"] = tgt_lang

        return batch


class WhisperModelModule(LightningModule):
    def __init__(self, cfg, joined_dataset: dict, sep_dataset: dict) -> None:
        super().__init__()

        self.model = Whisper.load_model(
            cfg.whisper_name,
            device='cpu',
            download_root=cfg.cache_dir, )
        if cfg.use_acti_ckpt:
            self.model.enable_acti_ckpt()

        self.tokenizer = WhisperTokenizer.from_pretrained('openai/whisper-large-v2', language="spanish",
                                                          cache_dir=cfg.cache_dir,
                                                          task='translate', predict_timestamps=False)

        self.loss_fn = nn.CrossEntropyLoss(ignore_index=-100)

        self.cfg = cfg
        self.__train_dataset = joined_dataset.get("train", [])
        self.__eval_dataset = sep_dataset.get("dev", [])
        self.__test_dataset = sep_dataset.get("test", [])
        self.decode_options = DecodingOptions(task='translate', beam_size=5, without_timestamps=True)

        self.valid_metrics = nn.ModuleDict(
            {"bleu": nn.ModuleList([torchmetrics.BLEUScore(compute_on_step=False) for _ in self.__eval_dataset]),
             "loss": nn.ModuleList([torchmetrics.MeanMetric(compute_on_step=False) for _ in self.__eval_dataset])})

        self.test_metrics = nn.ModuleDict(
            {"bleu": nn.ModuleList([torchmetrics.BLEUScore(compute_on_step=False) for _ in self.__test_dataset])})
        self.segment_tokenizers = get_segment_tokenizers()

    def forward(self, x):
        return self.model(x)

    def decode(self, audio_features, labels, src_langs, tgt_lang_codes, dataloader_idx, metrics):
        labels[labels == -100] = self.tokenizer.eos_token_id

        decode_res = decode(self.model.decoder, enc_hidden_states=audio_features,
                            lang_list=src_langs, options=self.decode_options)

        o_list = [res.text for res in decode_res]
        l_list = self.tokenizer.batch_decode(labels, skip_special_tokens=True)
        o_list_ = o_list.copy()
        l_list_ = l_list.copy()
        preprocess_sentence(o_list, tgt_lang_codes, self.segment_tokenizers)
        preprocess_sentence(l_list, tgt_lang_codes, self.segment_tokenizers)
        metrics['bleu'][dataloader_idx](o_list, [[l] for l in l_list])
        result = {
            'o_list': o_list_,
            'l_list': l_list_,
        }

        return result

    def training_step(self, batch, batch_id):
        input_ids = batch["input_ids"]
        labels = batch["labels"].long()
        dec_input_ids = batch["dec_input_ids"].long()

        audio_features = self.model.encoder(input_ids)

        out = self.model.decoder(dec_input_ids, audio_features)
        loss = self.loss_fn(out.view(-1, out.size(-1)), labels.view(-1))
        self.log("train/loss", loss, on_step=True, prog_bar=True, logger=True, on_epoch=True)
        return loss

    def on_validation_epoch_start(self) -> None:
        for k, v in self.valid_metrics.items():
            for metric in v:
                metric.set_dtype(torch.float32)

    def validation_step(self, batch, batch_id, dataloader_idx):
        input_ids = batch["input_ids"]
        labels = batch["labels"].long()
        dec_input_ids = batch["dec_input_ids"].long()

        audio_features = self.model.encoder(input_ids)
        logits = self.model.decoder(dec_input_ids, audio_features)

        loss = self.loss_fn(logits.view(-1, logits.size(-1)), labels.view(-1))

        result = self.decode(audio_features, labels, batch["src_lang"], batch["tgt_lang"], dataloader_idx,
                             self.valid_metrics)
        self.valid_metrics['loss'][dataloader_idx](loss)

        return {
            "loss": loss,
            "result": result,
        }

    def validation_epoch_end(self, outputs):
        loss_scores = [l.compute() for l in self.valid_metrics['loss']]
        self.log('valid_loss_epoch', torch.mean(torch.tensor(loss_scores)))
        print("valid_loss_epoch", torch.mean(torch.tensor(loss_scores)))
        bleu_scores = [b.compute() * 100 for b in self.valid_metrics['bleu']]
        self.log('valid_bleu_epoch', torch.mean(torch.tensor(bleu_scores)))
        print("valid_bleu_epoch", torch.mean(torch.tensor(bleu_scores)))
        for metrics in self.valid_metrics.values():
            for metric in metrics:
                metric.reset()

    def on_test_epoch_start(self) -> None:
        for k, v in self.test_metrics.items():
            for metric in v:
                metric.set_dtype(torch.float32)

    def test_step(self, batch, batch_id, dataloader_idx):
        input_ids = batch["input_ids"]
        labels = batch["labels"].long()

        audio_features = self.model.encoder(input_ids)

        result = self.decode(audio_features, labels, batch["src_lang"], batch["tgt_lang"], dataloader_idx,
                             self.test_metrics)

        return {
            "result": result,
        }

    def test_epoch_end(self, outputs):
        bleu_scores = [b.compute() * 100 for b in self.test_metrics['bleu']]
        for i, bleu in enumerate(bleu_scores):
            self.log(f"test_bleu_{i}", round(bleu.item(), 2))
            print(f"test_bleu_{i}", round(bleu.item(), 2))
        self.log('test_bleu_epoch', torch.mean(torch.tensor(bleu_scores)))
        print("test_bleu_epoch", torch.mean(torch.tensor(bleu_scores)))
        for metrics in self.test_metrics.values():
            for metric in metrics:
                metric.reset()

    def configure_optimizers(self):
        optimizer, scheduler = configure_optimizer_schedular(
            cfg=self.cfg,
            params_generator=self.named_parameters,
            num_training_steps=self.trainer.estimated_stepping_batches
        )
        self.optimizer = optimizer
        self.scheduler = scheduler

        return [optimizer], [{"scheduler": scheduler, "interval": "step", "frequency": 1}]

    def train_dataloader(self):
        dataset = WhisperTranslateDataset(self.__train_dataset, self.tokenizer, self.cfg.sample_rate, )
        return DataLoader(dataset,
                          batch_size=self.cfg.batch_size,
                          drop_last=True, shuffle=True, num_workers=self.cfg.num_worker,
                          collate_fn=WhisperDataCollatorWhithPadding(self.cfg)
                          )

    def val_dataloader(self):
        datasets = [WhisperTranslateDataset(dataset, self.tokenizer, self.cfg.sample_rate, )
                    for dataset in self.__eval_dataset]
        return [DataLoader(dataset,
                           batch_size=self.cfg.test_batch_size,
                           num_workers=self.cfg.num_worker,
                           collate_fn=WhisperDataCollatorWhithPadding(self.cfg)
                           ) for dataset in datasets]

    def test_dataloader(self):
        datasets = [WhisperTranslateDataset(dataset, self.tokenizer, self.cfg.sample_rate, )
                    for dataset in self.__test_dataset]
        return [DataLoader(dataset,
                           batch_size=self.cfg.test_batch_size,
                           num_workers=self.cfg.num_worker,
                           collate_fn=WhisperDataCollatorWhithPadding(self.cfg)
                           ) for dataset in datasets]


if __name__ == "__main__":
    from config.parse_yaml_args import parse_args_and_yaml
    from model.model_util import deep_to_device

    pd.options.display.max_rows = 100
    pd.options.display.max_colwidth = 1000

    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    cfg = parse_args_and_yaml(config_path="../config/exp_spec/whisper.yaml")
    DATA_ROOT = cfg.data_root
    language_list = cfg.language_list

    joined_data_pair_lists, sep_data_pair_lists = {}, {}
    for split in ["train", "dev", "test"]:
        joined_data_pair_lists[split], sep_data_pair_lists[split] = load_data_record(DATA_ROOT, split,
                                                                                     language_list=language_list, )

    cfg.batch_size = cfg.test_batch_size = 10
    cfg.num_worker = 0

    module = WhisperModelModule(cfg, joined_data_pair_lists, sep_data_pair_lists).to(cfg.device).eval()

    loader = module.test_dataloader()[0]

    with torch.no_grad():
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
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
