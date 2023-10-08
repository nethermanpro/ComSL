import os
import pandas as pd
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
from data.data_util import load_data_record, pad_trim_audio
from data.dataset import CascadeDataset
from model.model_util import load_mbart_model, load_mbart_tokenizer
from criterion.metric_util import get_segment_tokenizers, preprocess_sentence
from decode.whisper_decode import decode, DecodingOptions
from decode.mbart_decode import decode, DecodingOptions

pd.options.display.max_rows = 100
pd.options.display.max_colwidth = 1000


class CascadeDataCollatorWhithPadding:
    def __init__(self, cfg):
        self.cfg = cfg

    def __call__(self, features):
        audio, translations, transcription, audio_paths, src_langs, tgt_lang, m_src_langs = [], [], [], [], [], [], []
        for f in features:
            audio.append(f["audio"])
            translations.append(f["translation"])
            transcription.append(f["transcription"])
            audio_paths.append(f["audio_path"])
            src_langs.append(f["src_lang"])
            tgt_lang.append(f["tgt_lang"])
            m_src_langs.append(f["m_src_lang"])

        audio_input_feature = pad_trim_audio(audio, self.cfg)

        batch = {}

        batch["input_ids"] = audio_input_feature
        batch["audio_paths"] = audio_paths
        batch["src_langs"] = src_langs
        batch["translations"] = translations
        batch["transcription"] = transcription
        batch["m_src_langs"] = m_src_langs
        batch["tgt_lang"] = tgt_lang

        return batch


class CascadeModelModule(LightningModule):
    def __init__(self, cfg, joined_dataset: dict, sep_dataset: dict) -> None:
        super().__init__()
        model_name = cfg.model_name
        self.asr_model = Whisper.load_model(model_name, download_root=cfg.cache_dir, device='cpu')

        path = os.path.join(cfg.cache_dir, 'whisper_tokenizer')
        self.asr_tokenizer = WhisperTokenizer.from_pretrained(path, language="spanish", cache_dir=cfg.cache_dir,
                                                              task='transcribe', predict_timestamps=False)

        self.mt_tokenizer = load_mbart_tokenizer(cfg)
        self.mt_model = load_mbart_model(cfg, load_from_local=True, path=cfg.mbart_model_path).eval()

        self.cfg = cfg
        self.__train_dataset = joined_dataset.get("train", [])
        self.__eval_dataset = joined_dataset.get("dev", [])
        self.__test_dataset = sep_dataset.get("test", [])
        self.decode_options = DecodingOptions(task='transcribe', beam_size=5, without_timestamps=True)

        asr_state_dict = torch.load(f'{cfg.cache_dir}/{cfg.asr_model_path}', map_location=self.device)

        self.asr_model.load_state_dict(asr_state_dict)

        self.test_metrics = nn.ModuleDict(
            {"bleu": nn.ModuleList([torchmetrics.BLEUScore(compute_on_step=False) for _ in self.__test_dataset])})
        self.segment_tokenizers = get_segment_tokenizers()

    def forward(self, x):
        return self.asr_model(x)

    def training_step(self, batch, batch_id):
        pass

    def validation_step(self, batch, batch_id):
        pass

    def on_test_epoch_start(self) -> None:
        for k, v in self.test_metrics.items():
            for metric in v:
                metric.set_dtype(torch.float32)

    def test_step(self, batch, batch_id, dataloader_idx):
        input_ids = batch["input_ids"]
        src_langs = batch["src_langs"]
        tgt_lang = batch["tgt_lang"]
        m_src_lang = batch["m_src_langs"][0]
        audio_features = self.asr_model.encoder(input_ids)

        decode_res = decode(self.asr_model.decoder, enc_hidden_states=audio_features,
                            lang_list=src_langs, options=self.decode_options)

        asr_list = [res.text for res in decode_res]

        self.mt_tokenizer.src_lang = m_src_lang
        mt_imput_ids = self.mt_tokenizer(asr_list, return_tensors="pt", padding=True, truncation=True).input_ids.to(
            self.device)
        mt_gens = self.mt_model.generate(mt_imput_ids, forced_bos_token_id=self.mt_tokenizer.lang_code_to_id["en_XX"],
                                         max_new_tokens=100, num_beams=5)

        o_list = self.mt_tokenizer.batch_decode(mt_gens, skip_special_tokens=True)
        l_list = batch["translations"]
        preprocess_sentence(o_list, tgt_lang, self.segment_tokenizers)
        preprocess_sentence(l_list, tgt_lang, self.segment_tokenizers)
        self.test_metrics['bleu'][dataloader_idx](o_list, [[l] for l in l_list])

        return {
            'asr_list': asr_list,
            'asr_label': batch["transcription"],
            'o_list': o_list,
            'l_list': l_list,
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
        pass

    def train_dataloader(self):
        return None

    def val_dataloader(self):
        return None

    def test_dataloader(self):
        datasets = [CascadeDataset(test_dataset, self.asr_tokenizer, self.cfg.sample_rate) for test_dataset in
                    self.__test_dataset]
        return [torch.utils.data.DataLoader(dataset,
                                            batch_size=self.cfg.test_batch_size,
                                            num_workers=self.cfg.num_worker,
                                            collate_fn=CascadeDataCollatorWhithPadding(self.cfg)
                                            ) for dataset in datasets]


if __name__ == "__main__":
    from config.parse_yaml_args import parse_args_and_yaml
    from model.model_util import deep_to_device

    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    cfg = parse_args_and_yaml(config_path="../config/exp_spec/cascade.yaml")
    DATA_ROOT = cfg.data_root
    cfg.batch_size = cfg.test_batch_size = 10
    language_list = cfg.language_list

    joined_data_pair_lists, sep_data_pair_lists = {}, {}
    for split in ["train", "dev", "test"]:
        joined_data_pair_lists[split], sep_data_pair_lists[split] = load_data_record(DATA_ROOT, split,
                                                                                     language_list=language_list)
    module = CascadeModelModule(cfg, joined_data_pair_lists, sep_data_pair_lists).cuda().eval().half()

    loader = module.test_dataloader()[0]

    with torch.no_grad():
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            for b in loader:
                b = deep_to_device(b, "cuda")

                test_res = module.test_step(b, 0, 0)
                print(test_res)
                module.test_epoch_end([test_res])

                break
