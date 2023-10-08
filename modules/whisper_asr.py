import os
import numpy as np

import torchmetrics
import torch
from torch import nn
from torch.utils.data import DataLoader

from pytorch_lightning import LightningModule
from transformers import WhisperTokenizer

if __name__ == "__main__":
    import sys

    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.append(BASE_DIR)

import Whisper
from Whisper.normalizers import BasicTextNormalizer
from decode.whisper_decode import decode, DecodingOptions
from data.data_util import LANG_DICT, load_data_record
from data.data_util import pad_trim_audio
from data.dataset import CoVoSTDataset
from model.optimizer import configure_optimizer_schedular
from criterion.metric_util import get_segment_tokenizers, preprocess_sentence


class WhisperAsrDataset(CoVoSTDataset):
    def __init__(self, audio_info_list, tokenizer, sample_rate=16000) -> None:
        super().__init__(audio_info_list, tokenizer, sample_rate)

    def tokenize(self, src_lang, tgt_lang, transcription, translation):
        self.tokenizer.set_prefix_tokens(language=src_lang)
        transcription_ids = self.tokenizer(text=transcription, return_tensors="np", max_length=self.max_text_length,
                                           truncation=True).input_ids[0].tolist()
        transcription_labels = transcription_ids[1:] + [self.tokenizer.eos_token_id]
        return transcription_ids, None, transcription_labels, None


class WhisperAsrDataCollatorWhithPadding:
    def __init__(self, cfg, pad_token_id=-100):
        self.cfg = cfg
        self.pad_token_id = pad_token_id


    def __call__(self, features):
        audio, labels, dec_input_ids, audio_paths, src_langs, tgt_langs = [], [], [], [], [], []
        for f in features:
            audio.append(f["audio"])
            labels.append(f["transcription_labels"])
            dec_input_ids.append(f["transcription_ids"])
            audio_paths.append(f["audio_path"])
            src_langs.append(LANG_DICT[f["src_lang"]]['whisper'])
            tgt_langs.append(LANG_DICT[f["tgt_lang"]]['whisper'])

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
        batch["src_langs"] = src_langs
        batch["tgt_langs"] = tgt_langs

        return batch


class WhisperAsrModelModule(LightningModule):
    def __init__(self, cfg, joined_dataset: dict, sep_dataset: dict) -> None:
        super().__init__()
        self.model = Whisper.load_model(
            cfg.whisper_name,
            device='cpu',
            download_root=cfg.cache_dir,
        )
        self.tokenizer = WhisperTokenizer.from_pretrained('openai/whisper-large-v2', language="spanish",
                                                          cache_dir=cfg.cache_dir,
                                                          task='transcribe', predict_timestamps=False)
        self.loss_fn = nn.CrossEntropyLoss(ignore_index=-100)

        self.cfg = cfg
        self.__train_dataset = joined_dataset.get("train", [])
        self.__eval_dataset = sep_dataset.get("dev", [])
        self.__test_dataset = sep_dataset.get("test", [])
        self.decode_options = DecodingOptions(task='transcribe', beam_size=5, without_timestamps=True)

        self.normalizer = BasicTextNormalizer()
        if cfg.use_acti_ckpt:
            self.model.enable_acti_ckpt()

        self.valid_metrics = nn.ModuleDict({
            "loss": nn.ModuleList([torchmetrics.MeanMetric(compute_on_step=False) for _ in self.__eval_dataset]),
            'wer': nn.ModuleList([torchmetrics.WordErrorRate(compute_on_step=False) for _ in self.__eval_dataset]),
        })

        self.test_metrics = nn.ModuleDict({
            'wer': nn.ModuleList([torchmetrics.WordErrorRate(compute_on_step=False) for _ in self.__test_dataset]),
        })

        self.segment_tokenizers = get_segment_tokenizers()

    def forward(self, x):
        return self.model(x)

    def decode(self, audio_features, labels, src_langs, tgt_langs, metrics, dataloader_idx):
        labels[labels == -100] = self.tokenizer.eos_token_id

        decode_res = decode(self.model.decoder, enc_hidden_states=audio_features,
                            lang_list=src_langs, options=self.decode_options)

        o_list = [res.text for res in decode_res]
        l_list = self.tokenizer.batch_decode(labels, skip_special_tokens=True)

        preprocess_sentence(o_list, src_langs, self.segment_tokenizers)
        preprocess_sentence(l_list, src_langs, self.segment_tokenizers)

        o_list_ = [self.normalizer(o) for o in o_list]
        l_list_ = [self.normalizer(l) for l in l_list]

        metrics["wer"][dataloader_idx](o_list_, l_list_)
        result = {
            'st_res': o_list_,
            'st_ref': l_list_,
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

    def validation_step(self, batch, batch_id, dataloader_idx=None):
        if dataloader_idx is None:
            print("warning: dataloader_idx is None")
            dataloader_idx = 0
        input_ids = batch["input_ids"]
        labels = batch["labels"].long()
        dec_input_ids = batch["dec_input_ids"].long()

        audio_features = self.model.encoder(input_ids)
        out = self.model.decoder(dec_input_ids, audio_features)

        loss = self.loss_fn(out.view(-1, out.size(-1)), labels.view(-1))

        result = self.decode(audio_features, labels, batch["src_langs"], batch["tgt_langs"], self.valid_metrics,
                             dataloader_idx)

        self.valid_metrics['loss'][dataloader_idx](loss)

        return {
            "loss": loss,
            "result": result
        }

    def validation_epoch_end(self, outputs):
        loss_scores = [l.compute() for l in self.valid_metrics['loss']]
        self.log('valid_loss_epoch', torch.mean(torch.tensor(loss_scores)))
        print("valid_loss_epoch", torch.mean(torch.tensor(loss_scores)))
        wer_scores = [b.compute() for b in self.valid_metrics['wer']]
        self.log('valid_wer_epoch', torch.mean(torch.tensor(wer_scores)))
        print("valid_wer_epoch", torch.mean(torch.tensor(wer_scores)))
        for metrics in self.valid_metrics.values():
            for metric in metrics:
                metric.reset()

    def on_test_epoch_start(self) -> None:
        for k, v in self.test_metrics.items():
            for metric in v:
                metric.set_dtype(torch.float32)

    def test_step(self, batch, batch_id, dataloader_idx=None):
        if dataloader_idx is None:
            print("warning: dataloader_idx is None")
            dataloader_idx = 0

        input_ids = batch["input_ids"]
        labels = batch["labels"].long()
        audio_features = self.model.encoder(input_ids)
        result = self.decode(audio_features, labels, batch["src_langs"], batch["tgt_langs"], self.test_metrics,
                             dataloader_idx)

        return {
            'result': result,
        }

    def test_epoch_end(self, outputs):
        wer_scores = [b.compute() for b in self.test_metrics['wer']]
        for i, wer in enumerate(wer_scores):
            self.log(f"test_wer_{i}", round(wer.item(), 2))
            print(f"test_wer_{i}", round(wer.item(), 2))
        self.log('test_wer_epoch', torch.mean(torch.tensor(wer_scores)))
        print("test_wer_epoch", torch.mean(torch.tensor(wer_scores)))
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
        dataset = WhisperAsrDataset(self.__train_dataset, self.tokenizer, self.cfg.sample_rate)
        return DataLoader(dataset,
                          batch_size=self.cfg.batch_size,
                          drop_last=True, shuffle=True, num_workers=self.cfg.num_worker,
                          collate_fn=WhisperAsrDataCollatorWhithPadding(self.cfg)
                          )

    def val_dataloader(self):
        datasets = [WhisperAsrDataset(dataset, self.tokenizer, self.cfg.sample_rate) for dataset in self.__eval_dataset]
        return [DataLoader(dataset,
                           batch_size=self.cfg.test_batch_size,
                           num_workers=self.cfg.num_worker,
                           collate_fn=WhisperAsrDataCollatorWhithPadding(self.cfg)
                           ) for dataset in datasets]

    def test_dataloader(self):
        datasets = [WhisperAsrDataset(dataset, self.tokenizer, self.cfg.sample_rate) for dataset in self.__test_dataset]
        return [DataLoader(dataset,
                           batch_size=self.cfg.test_batch_size,
                           num_workers=self.cfg.num_worker,
                           collate_fn=WhisperAsrDataCollatorWhithPadding(self.cfg)
                           ) for dataset in datasets]


if __name__ == "__main__":
    from config.parse_yaml_args import parse_args_and_yaml

    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    cfg = parse_args_and_yaml(config_path="../config/exp_spec/whisper_asr.yaml")
    DATA_ROOT = cfg.data_root
    language_list = cfg.language_list

    joined_data_pair_lists, sep_data_pair_lists = {}, {}
    for split in ["train", "dev", "test"]:
        joined_data_pair_lists[split], sep_data_pair_lists[split] = load_data_record(DATA_ROOT, split,
                                                                                     language_list=language_list)

    cfg.batch_size = cfg.test_batch_size = 1

    module = WhisperAsrModelModule(cfg, joined_data_pair_lists, sep_data_pair_lists).to(cfg.device).eval()

    loader = module.test_dataloader()[0]
    asr_state_dict = torch.load(f'{cfg.cache_dir}/whisper_asr.pt', map_location='cuda')

    module.model.load_state_dict(asr_state_dict)


    def deep_to_device(obj, device):
        if isinstance(obj, torch.Tensor):
            return obj.to(device)
        elif isinstance(obj, dict):
            return {k: deep_to_device(v, device) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [deep_to_device(v, device) for v in obj]
        else:
            return obj


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
