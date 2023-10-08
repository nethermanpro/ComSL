import numpy as np
import torch
from torch import nn
import os
from pytorch_lightning import LightningModule
import torchmetrics
from torch.utils.data import DataLoader

if __name__ == "__main__":
    import sys

    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.append(BASE_DIR)

from data.data_util import LANG_DICT, load_data_record, pad_trim_audio
from data.dataset import ComSTDataset
from decode.mbart_decode import decode, DecodingOptions
from criterion.mix_criterions import GuidedCrossEntMultiTaskCriterion
from criterion.metric_util import get_segment_tokenizers, preprocess_sentence
from model.optimizer import configure_optimizer_schedular
from model.model_util import load_mbart_tokenizer
from model.ComSL_model import ComSTModel
from Whisper.normalizers import BasicTextNormalizer


class ComSTCollatorWhithPadding:
    def __init__(self, cfg, pad_token_id, tokenizer) -> None:
        self.pad_token_id = pad_token_id
        self.cfg = cfg
        self.tokenizer = tokenizer
        self.p_mask = cfg.p_mask

    def __call__(self, features):
        transcription_ids, translation_ids, transcription_labels, translation_labels, audio, src_lang_codes, tgt_lang_codes = [], [], [], [], [], [], []
        for f in features:
            transcription_ids.append(f["transcription_ids"])
            translation_ids.append(f["translation_ids"])
            transcription_labels.append(f["transcription_labels"])
            translation_labels.append(f["translation_labels"])
            audio.append(f["audio"])
            src_lang_codes.append(LANG_DICT[f['src_lang']]['whisper'])
            tgt_lang_codes.append(LANG_DICT[f['tgt_lang']]['whisper'])

        # audio
        mel = pad_trim_audio(audio, self.cfg)
        # transcription
        transcription_len = [len(t) for t in transcription_ids]
        max_transcription_len = max(transcription_len)

        src_txt_tokens = [
            np.pad(ids, (0, max_transcription_len - length), 'constant', constant_values=self.pad_token_id) for
            ids, length in zip(transcription_ids, transcription_len)]
        transcription_labels = [
            np.pad(ids, (0, max_transcription_len - length), 'constant', constant_values=self.pad_token_id) for
            ids, length in zip(transcription_labels, transcription_len)]

        # text encoder
        translation_len = [len(t) for t in translation_ids]
        max_translation_len = max(translation_len)

        tgt_txt_tokens = [np.pad(ids, (0, max_translation_len - length), 'constant', constant_values=self.pad_token_id)
                          for ids, length in zip(translation_ids, translation_len)]
        translation_labels = [
            np.pad(ids, (0, max_translation_len - length), 'constant', constant_values=self.pad_token_id) for
            ids, length in zip(translation_labels, translation_len)]

        def to_tensor(x):
            return torch.tensor(np.array(x), requires_grad=False)

        src_txt_tokens = to_tensor(src_txt_tokens).long()
        tgt_txt_tokens = to_tensor(tgt_txt_tokens).long()

        mlm_mask = torch.rand(src_txt_tokens.shape) < self.p_mask
        mlm_mask = mlm_mask & (src_txt_tokens != self.pad_token_id) & (src_txt_tokens != 2)
        mlm_mask[:, 0] = False

        while mlm_mask.sum() == 0:
            mlm_mask = torch.rand(src_txt_tokens.shape) < self.p_mask
            mlm_mask = mlm_mask & (src_txt_tokens != self.pad_token_id) & (src_txt_tokens != 2)
            mlm_mask[:, 0] = False

        masked_src_tokens = src_txt_tokens.clone()
        masked_src_tokens[mlm_mask] = self.tokenizer.mask_token_id

        tgt_mlm_mask = torch.roll(mlm_mask, -1, dims=1)

        src_lang_ids = src_txt_tokens[:, 0]

        batch = {
            "net_input": {
                "mel": mel,
                "src_lang_ids": src_lang_ids,
                "tokens": [src_txt_tokens, tgt_txt_tokens],
                'masked_src_tokens': masked_src_tokens,
                'mlm_mask': tgt_mlm_mask,
                "txt_lengths": [to_tensor(transcription_len), to_tensor(translation_len)],
            },
            "target": [to_tensor(transcription_labels).long(), to_tensor(translation_labels).long()],
            "dec_start_ids": [src_txt_tokens[:, 0], tgt_txt_tokens[:, 0]],
            "ntokens": [sum(transcription_len), sum(translation_len)],
            "tgt_lang_codes": tgt_lang_codes,
            "src_lang_codes": src_lang_codes,
        }

        return batch


class ComSTModule(LightningModule):
    def __init__(self, cfg, joined_dataset: dict, sep_dataset: dict) -> None:
        super().__init__()

        self.tokenizer = load_mbart_tokenizer(cfg)
        self.model = ComSTModel(cfg)
        self.decode_options = DecodingOptions(beam_size=5)
        self.collect_fn = ComSTCollatorWhithPadding(cfg, self.tokenizer.pad_token_id, tokenizer=self.tokenizer)
        self.automatic_optimization = True

        self.cfg = cfg
        self.__train_dataset = joined_dataset.get("train", [])
        self.__eval_dataset = sep_dataset.get("dev", [])
        self.__test_dataset = sep_dataset.get("test", [])

        self.segment_tokenizers = get_segment_tokenizers()
        self.normalizer = BasicTextNormalizer()

        self.train_criterion = GuidedCrossEntMultiTaskCriterion(cfg, self.tokenizer.pad_token_id)
        self.val_criterion = nn.ModuleList(
            [GuidedCrossEntMultiTaskCriterion(cfg, self.tokenizer.pad_token_id) for _ in self.__eval_dataset])
        self.valid_metrics = nn.ModuleDict({
            'bleu_spch': nn.ModuleList([torchmetrics.BLEUScore(compute_on_step=False) for _ in self.__eval_dataset]),
            'bleu_text': nn.ModuleList([torchmetrics.BLEUScore(compute_on_step=False) for _ in self.__eval_dataset]),
            'wer': nn.ModuleList([torchmetrics.WordErrorRate(compute_on_step=False) for _ in self.__eval_dataset]),
        })
        self.test_metrics = nn.ModuleDict({
            'bleu_spch': nn.ModuleList([torchmetrics.BLEUScore(compute_on_step=False) for _ in self.__test_dataset]),
            'bleu_text': nn.ModuleList([torchmetrics.BLEUScore(compute_on_step=False) for _ in self.__test_dataset]),
            'wer': nn.ModuleList([torchmetrics.WordErrorRate(compute_on_step=False) for _ in self.__test_dataset]),
        })

    def forward(self, **kwargs):
        return self.model(**kwargs)

    def training_step(self, batch, batch_id):

        self.model.set_num_updates(self.global_step, self.current_epoch)
        loss, logging_output, net_output = self.train_criterion(self.model, batch)
        for log_item, data in logging_output.items():
            self.log(f"train_{log_item}", data, on_step=True, prog_bar=True, logger=True, on_epoch=False,
                     sync_dist=True)
        return loss

    def on_validation_epoch_start(self) -> None:
        for k, v in self.valid_metrics.items():
            for metric in v:
                metric.set_dtype(torch.float32)
        for criterion in self.val_criterion:
            for metric in criterion.metrics.values():
                metric.set_dtype(torch.float32)

    def validation_step(self, batch, batch_id, dataloader_idx=None):
        if dataloader_idx is None:
            print("warning: dataloader_idx is None")
            dataloader_idx = 0
        labels = batch["target"]
        loss, logging_output, net_output = self.val_criterion[dataloader_idx](self.model, batch)

        enc_out = net_output[1]
        hidden_states = [enc_out[0]['encoder_out'], enc_out[1]['encoder_out']]
        decode_res = self.decode(hidden_states,
                                 labels,
                                 batch["dec_start_ids"],
                                 batch["src_lang_codes"],
                                 batch["tgt_lang_codes"],
                                 dataloader_idx,
                                 self.valid_metrics)

        return {
            "result": decode_res,
            "logging_output": logging_output
        }

    def validation_epoch_end(self, outputs):
        for metric_name, metric_list in self.valid_metrics.items():
            results = [m.compute() for m in metric_list]
            if 'bleu' in metric_name:
                results = [r * 100 for r in results]
            mean_result = sum(results) / len(results)
            self.log(f"val_{metric_name}_epoch", mean_result)
            print(f"val_{metric_name}_epoch", mean_result)
            for m in metric_list:
                m.reset()
        valid_reports = {}
        for criterion in self.val_criterion:
            valid_report = criterion.reduce_metric()
            for k, v in valid_report.items():
                if k not in valid_reports:
                    valid_reports[k] = v
                else:
                    valid_reports[k] += v
        average_valid_reports = {k: v / len(self.val_criterion) for k, v in valid_reports.items()}
        for k, v in average_valid_reports.items():
            self.log(f"val_{k}_epoch", v)
            print(f"val_{k}_epoch", v)

    def on_test_epoch_start(self) -> None:
        for metric_name, metric_list in self.test_metrics.items():
            for m in metric_list:
                m.set_dtype(torch.float32)

    def test_step(self, batch, batch_idx, dataloader_idx=None):
        if dataloader_idx is None:
            print("warning: dataloader_idx is None")
            dataloader_idx = 0
        labels = batch["target"]
        net_input = batch["net_input"]
        enc_out = self.model.encoder(
            mel=net_input["mel"],
            src_tokens=net_input["tokens"][0],
            src_lang_ids=net_input["src_lang_ids"],
            masked_src_tokens=None,
        )

        hidden_states = [enc_out[0]['encoder_out'], enc_out[1]['encoder_out']]
        decode_res = self.decode(hidden_states,
                                 labels,
                                 batch["dec_start_ids"],
                                 batch["src_lang_codes"],
                                 batch["tgt_lang_codes"],
                                 dataloader_idx,
                                 self.test_metrics)

        return decode_res

    def test_epoch_end(self, outputs):
        for metric_name, metric_list in self.test_metrics.items():
            results = [m.compute() for m in metric_list]
            if 'bleu' in metric_name:
                results = [r * 100 for r in results]
            mean_result = sum(results) / len(results)
            self.log(f"test_{metric_name}_epoch", mean_result)
            for i, r in enumerate(results):
                self.log(f"test_{metric_name}_{i}", r)
            print(f"test_{metric_name}_epoch", mean_result)
            for m in metric_list:
                m.reset()

    def decode(self, hidden_states, labels, decoder_start_ids, src_lang_codes, tgt_lang_codes, dataloader_idx, metrics):

        transcript_start_ids, translate_start_ids = decoder_start_ids
        transcript_labels, translate_labels = labels
        spch_hiddens, text_hiddens = hidden_states

        labels = {'transcript': transcript_labels, 'translate': translate_labels}
        start_ids = {'transcript': transcript_start_ids, 'translate': translate_start_ids}
        result = {}
        for task in ['transcript', 'translate']:
            if labels[task] is None:
                continue
            spch_pred_token = decode(self.model.decoder,
                                     self.tokenizer,
                                     enc_hidden_states=spch_hiddens,
                                     forced_bos_token_id=start_ids[task],
                                     options=self.decode_options, )

            spch_detoken_out = self.tokenizer.batch_decode(spch_pred_token, skip_special_tokens=True)
            detoken_label = self.tokenizer.batch_decode(labels[task], skip_special_tokens=True)
            lang_codes = src_lang_codes if task == 'transcript' else tgt_lang_codes
            preprocess_sentence(spch_detoken_out, lang_codes, self.segment_tokenizers)
            preprocess_sentence(detoken_label, lang_codes, self.segment_tokenizers)
            if task == 'translate':
                text_pred_token = decode(self.model.decoder,
                                         self.tokenizer,
                                         enc_hidden_states=text_hiddens,
                                         forced_bos_token_id=start_ids[task],
                                         options=self.decode_options, )
                text_detoken_out = self.tokenizer.batch_decode(text_pred_token, skip_special_tokens=True)
                preprocess_sentence(text_detoken_out, lang_codes, self.segment_tokenizers)
                result[f'{task}_text_pred'] = text_detoken_out
            if task == 'translate':
                metrics['bleu_spch'][dataloader_idx](spch_detoken_out, [[l] for l in detoken_label])
                metrics['bleu_text'][dataloader_idx](text_detoken_out, [[l] for l in detoken_label])
            elif task == 'transcript':
                o_list_ = [self.normalizer(o) for o in spch_detoken_out]
                l_list_ = [self.normalizer(l) for l in detoken_label]
                metrics['wer'][dataloader_idx](o_list_, l_list_)
            result[f'{task}_spch_pred'] = spch_detoken_out
            result[f'{task}_label'] = detoken_label

        return result

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
        dataset = ComSTDataset(self.__train_dataset, self.tokenizer, self.cfg.sample_rate, cfg=self.cfg)
        return DataLoader(dataset,
                          batch_size=self.cfg.batch_size,
                          drop_last=True, shuffle=True, num_workers=self.cfg.num_worker,
                          collate_fn=self.collect_fn
                          )

    def val_dataloader(self):
        datasets = [ComSTDataset(dataset, self.tokenizer, self.cfg.sample_rate, cfg=self.cfg) for
                    dataset in self.__eval_dataset]
        return [DataLoader(dataset,
                           batch_size=self.cfg.test_batch_size,
                           num_workers=self.cfg.num_worker,
                           collate_fn=self.collect_fn
                           ) for dataset in datasets]

    def test_dataloader(self):
        datasets = [ComSTDataset(dataset, self.tokenizer, self.cfg.sample_rate, cfg=self.cfg) for
                    dataset in self.__test_dataset]
        return [DataLoader(dataset,
                           batch_size=self.cfg.test_batch_size,
                           num_workers=self.cfg.num_worker,
                           collate_fn=self.collect_fn
                           ) for dataset in datasets]


if __name__ == "__main__":
    from config.parse_yaml_args import parse_args_and_yaml
    from model.model_util import deep_to_device

    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    cfg = parse_args_and_yaml(config_path="../config/exp_spec/comsl.yaml")

    DATA_ROOT = cfg.data_root
    language_list = cfg.language_list
    extra_language_list = cfg.extra_language_list

    joined_data_pair_lists, sep_data_pair_lists = {}, {}
    for split in ["train", "dev", "test"]:
        joined_data_pair_lists[split], sep_data_pair_lists[split] = load_data_record(
            DATA_ROOT,
            split,
            language_list=language_list,
            expanded_data_root=cfg.cv_data_root,
            expanded_language_list=extra_language_list)
    if "OUTPUT_DIR" in os.environ:
        output_dir = os.environ["OUTPUT_DIR"]
    else:
        output_dir = cfg.output_dir

    cfg.cache_dir = f"{output_dir}/cache"

    cfg.batch_size = cfg.test_batch_size = 10
    cfg.num_worker = 4

    module = ComSTModule(cfg, joined_data_pair_lists, sep_data_pair_lists).cuda().eval()

    loader = module.test_dataloader()[0]

    optimizer, scheduler = configure_optimizer_schedular(
        cfg=module.cfg,
        params_generator=module.named_parameters,
        num_training_steps=10000
    )

    with torch.no_grad():
        for b in loader:

            b = deep_to_device(b, 'cuda')
            train_res = module.training_step(b, 0)
            print(train_res)

            valid_res = module.validation_step(b, 0, 0)
            print(valid_res)
            module.validation_epoch_end([valid_res])

            test_res = module.test_step(b, 0, 0)
            print(test_res)
            module.test_epoch_end([test_res])

            break
