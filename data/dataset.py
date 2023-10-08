import os
import torch
from data.data_util import load_wave, LANG_DICT
from torch.utils.data import Dataset


class CoVoSTDataset(Dataset):
    def __init__(self, audio_info_list, tokenizer, sample_rate=16000) -> None:
        super().__init__()

        self.audio_info_list = audio_info_list
        self.tokenizer = tokenizer
        self.sample_rate = sample_rate
        self.max_text_length = 128

    def __len__(self):
        return len(self.audio_info_list)

    def __getitem__(self, id):
        record = self.audio_info_list[id]

        audio_path = os.path.join(record['audio_root'], record["path"])

        src_lang = record["src_lang"]
        tgt_lang = record["tgt_lang"]

        transcription = record["sentence"]
        translation = record["translation"]

        # # text
        tokenize_res = self.tokenize(src_lang, tgt_lang, transcription, translation)

        # audio
        audio, duration = load_wave(audio_path, sample_rate=self.sample_rate)
        audio = audio.flatten()

        return {
            'index': id,
            "transcription_ids": tokenize_res[0],
            "translation_ids": tokenize_res[1],
            "transcription_labels": tokenize_res[2],
            "translation_labels": tokenize_res[3],
            "audio": audio,
            "duration": duration,
            "src_lang": src_lang,
            "tgt_lang": tgt_lang,
            "audio_path": audio_path,
        }

    def tokenize(self, src_lang, tgt_lang, transcription, translation):
        raise NotImplementedError


class ComSTDataset(torch.utils.data.Dataset):
    def __init__(self, audio_info_list, tokenizer, sample_rate=16000, cfg=None) -> None:
        super().__init__()

        self.audio_info_list = audio_info_list
        self.tokenizer = tokenizer
        self.sample_rate = sample_rate
        self.max_text_length = 128
        self.cfg = cfg

    def __len__(self):
        return len(self.audio_info_list)

    def __getitem__(self, id):
        record = self.audio_info_list[id]

        audio_path = os.path.join(record['audio_root'], record["path"])

        src_lang = record["src_lang"]
        tgt_lang = record["tgt_lang"]

        transcription = record["sentence"]
        translation = record["translation"]

        # text
        tokenize_res = self.tokenize(src_lang, tgt_lang, transcription, translation)

        # audio
        audio, duration = load_wave(audio_path, sample_rate=self.sample_rate)
        audio = audio.flatten()

        return {
            'index': id,
            "transcription_ids": tokenize_res[0],
            "translation_ids": tokenize_res[1],
            "transcription_labels": tokenize_res[2],
            "translation_labels": tokenize_res[3],
            "audio": audio,
            "duration": duration,
            "src_lang": src_lang,
            "tgt_lang": tgt_lang,
            "audio_path": audio_path,
        }

    def tokenize(self, src_lang, tgt_lang, transcription, translation):
        # text
        self.tokenizer.src_lang = LANG_DICT[src_lang]['mbart']
        self.tokenizer.tgt_lang = LANG_DICT[tgt_lang]['mbart']

        tokenize_res = self.tokenizer(transcription, text_target=translation, max_length=self.max_text_length,
                                      truncation=True, return_tensors="np")
        transcription_ids = tokenize_res['input_ids'][0].tolist()
        translation_ids = tokenize_res['labels'][0].tolist()

        transcription_labels = transcription_ids[1:] + [self.tokenizer.pad_token_id]
        translation_labels = translation_ids[1:] + [self.tokenizer.pad_token_id]
        return transcription_ids, translation_ids, transcription_labels, translation_labels


class CascadeDataset(Dataset):
    def __init__(self, audio_info_list, tokenizer, sample_rate, device=None) -> None:
        super().__init__()

        self.audio_info_list = audio_info_list
        self.sample_rate = sample_rate
        self.tokenizer = tokenizer
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def __len__(self):
        return len(self.audio_info_list)

    def __getitem__(self, id):
        record = self.audio_info_list[id]
        audio_path = os.path.join(record['audio_root'], record["path"])
        translation = record["translation"]
        transcription = record["sentence"]

        # audio
        audio, duration = load_wave(audio_path, sample_rate=self.sample_rate)
        audio = audio.flatten()

        return {
            "audio": audio,
            "audio_path": audio_path,
            "transcription": transcription,
            "translation": translation,
            "src_lang": LANG_DICT[record['src_lang']]['whisper'],
            "tgt_lang": LANG_DICT[record['tgt_lang']]['whisper'],
            "m_src_lang": LANG_DICT[record['src_lang']]['mbart'],
        }


class MbartDataset(Dataset):
    def __init__(self, audio_info_list, tokenizer) -> None:
        super().__init__()

        self.audio_info_list = audio_info_list
        self.tokenizer = tokenizer
        self.max_length = 128

    def __len__(self):
        return len(self.audio_info_list)

    def __getitem__(self, id):
        record = self.audio_info_list[id]

        translation = record["translation"]
        transcription = record["sentence"]
        src_lang = record["src_lang"]
        tgt_lang = record["tgt_lang"]

        self.tokenizer.src_lang = LANG_DICT[src_lang]['mbart']
        self.tokenizer.tgt_lang = LANG_DICT[tgt_lang]['mbart']

        encoded_ids = self.tokenizer(text=transcription, text_target=translation, max_length=self.max_length,
                                     truncation=True, return_tensors="np")
        encoded_src = encoded_ids['input_ids'][0].tolist()
        encoded_tgt = encoded_ids['labels'][0].tolist()

        dec_input_ids = [2] + encoded_tgt[:-1]

        return {
            "labels": encoded_tgt,
            "dec_input_ids": dec_input_ids,
            "enc_input_ids": encoded_src,
            "tgt_lang": LANG_DICT[tgt_lang]['whisper']
        }
