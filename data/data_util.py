import pandas as pd
import torchaudio
import torchaudio.transforms as at
import os
import torch
import Whisper

LANG_DICT = pd.read_csv('data/lang_dict.csv').set_index("full name").to_dict("index")


def read_table(path):
    return pd.read_table(path, on_bad_lines='error', quoting=3, doublequote=False, encoding='utf-8', engine="python")


def load_data_record(data_root, split, language_list, subsample_rate=1,
                     expanded_data_root=None, expanded_language_list=None):
    data_pair_lists = []
    for lang in language_list:
        data_lang_code = LANG_DICT[lang]['covost']
        data_pair = read_table(os.path.join(data_root, 'covost',
                                            f"{data_lang_code}_en", f"covost_v2.{data_lang_code}_en.{split}.tsv"))
        data_pair['src_lang'] = lang
        data_pair['tgt_lang'] = 'english'
        data_pair['audio_root'] = os.path.join(data_root, 'extracted', data_lang_code, 'clips')
        data_pair = data_pair.dropna()
        data_pair_lists.append(data_pair)
        print(f"Loaded {len(data_pair)} {lang} to english data pairs.")
        if split == 'train' and expanded_language_list is not None and lang in expanded_language_list:
            expanded_data_pair = read_table(os.path.join(expanded_data_root, 'psudo', f"{data_lang_code}_en.train.tsv"))
            test_data_pair = read_table(os.path.join(data_root, 'covost', f"{data_lang_code}_en",
                                                     f"covost_v2.{data_lang_code}_en.test.tsv"))
            dev_data_pair = read_table(os.path.join(data_root, 'covost', f"{data_lang_code}_en",
                                                    f"covost_v2.{data_lang_code}_en.dev.tsv"))
            expanded_data_pair = pd.concat(
                [expanded_data_pair, data_pair, data_pair, test_data_pair, test_data_pair, dev_data_pair,
                 dev_data_pair], ignore_index=True).drop_duplicates(subset=['path'], keep=False)
            expanded_data_pair['src_lang'] = lang
            expanded_data_pair['tgt_lang'] = 'english'
            expanded_data_pair['audio_root'] = os.path.join(expanded_data_root, data_lang_code, 'clips')
            data_pair_lists.append(expanded_data_pair)
            print(f"Loaded {len(expanded_data_pair)} {lang} to english extra data pairs.")
    joined_data_pair_lists = pd.concat(data_pair_lists, ignore_index=True).to_dict("records")[::subsample_rate]
    data_pair_lists = [data_pair_list.to_dict("records")[::subsample_rate] for data_pair_list in data_pair_lists]
    return joined_data_pair_lists, data_pair_lists


def load_wave(wave_path, sample_rate: int = 16000):
    waveform, sr = torchaudio.load(wave_path, normalize=True)

    duration = waveform.shape[1] / sr
    if sample_rate != sr:
        waveform = at.Resample(sr, sample_rate)(waveform)
    return waveform, duration


def pad_trim_audio(audio, cfg):
    max_lens = cfg.chunk_size * cfg.sample_rate
    audio_input_feature = [Whisper.log_mel_spectrogram(Whisper.pad_or_trim(a, max_lens)) for a in audio]
    audio_input_feature = torch.concat([a[None, :] for a in audio_input_feature])
    return audio_input_feature
