import torch
import os
import pandas as pd

from tqdm import tqdm
from torch.utils.data import DataLoader

from data.data_util import LANG_DICT
from config.parse_yaml_args import parse_args_and_yaml
from model.model_util import load_mbart_model, load_mbart_tokenizer

os.environ["TOKENIZERS_PARALLELISM"] = "false"


class MbartDataset(torch.utils.data.Dataset):
    def __init__(self, audio_info_list) -> None:
        super().__init__()

        self.audio_info_list = audio_info_list

    def __len__(self):
        return len(self.audio_info_list)

    def __getitem__(self, id):
        record = self.audio_info_list[id]

        sentence = record["sentence"]

        return {
            "sentence": sentence,
            "index": id,
        }


class MbartCollatorWhithPadding:
    def __init__(
        self,
        tokenizer,
        src_lang,
    ) -> None:
        super().__init__()
        self.tokenizer = tokenizer

        self.tokenizer.src_lang = LANG_DICT[src_lang]["mbart"]

    def __call__(self, features):
        (
            sentences,
            indexs,
        ) = (
            [],
            [],
        )
        for f in features:
            sentences.append(f["sentence"])
            indexs.append(f["index"])

        inputs = self.tokenizer(
            sentences, padding=True, truncation=True, return_tensors="pt"
        )

        batch = {
            "inputs": inputs,
            "indexs": indexs,
        }

        return batch


cfg = parse_args_and_yaml(config_path="config/exp_spec/mbart.yaml")

CV_root = ""  # TODO: set your Common Voice root path
data_language = "french"  # TODO: set your data language, e.g. 'french', 'chinese'
cfg.mbart_model_path = ""  # TODO: set your mBART model path to your pretrained model
output_dir = f"{CV_root}/pseudo"


mbart_tokenizer = load_mbart_tokenizer(cfg)
mbart_model = load_mbart_model(cfg)
mbart_model = mbart_model.to("cuda")

if __name__ == "__main__":
    data_code = LANG_DICT[data_language]["covost"]
    for split in ["train"]:
        if os.path.exists(os.path.join(output_dir, f"{data_code}_en.{split}.tsv")):
            continue
        table_path = os.path.join(CV_root, data_code, f"{split}.tsv")
        data_pair = pd.read_table(
            table_path,
            on_bad_lines="error",
            quoting=3,
            doublequote=False,
            encoding="utf-8",
            engine="python",
        )
        output_data_pair = data_pair.copy()[["path", "sentence"]]
        output_data_pair["translation"] = None
        data_pair_list = data_pair.to_dict("records")
        print(f"Loaded {len(data_pair)} {data_language} to english {split} data pairs.")

        dataset = MbartDataset(data_pair_list)
        collecter = MbartCollatorWhithPadding(mbart_tokenizer, src_lang=data_language)
        dataloader = DataLoader(
            dataset,
            batch_size=40,
            shuffle=False,
            collate_fn=collecter,
            num_workers=2,
            pin_memory=True,
            drop_last=False,
        )
        for batch in tqdm(dataloader):
            inputs = batch["inputs"].to("cuda")
            indexs = batch["indexs"]
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                with torch.no_grad():
                    outputs = mbart_model.generate(
                        **inputs,
                        decoder_start_token_id=mbart_tokenizer.lang_code_to_id["en_XX"],
                    )
            translations = mbart_tokenizer.batch_decode(
                outputs, skip_special_tokens=True
            )
            for i, index in enumerate(indexs):
                output_data_pair.loc[index, "translation"] = translations[i]
        output_data_pair.to_csv(
            os.path.join(output_dir, f"{data_code}_en.{split}.tsv"),
            sep="\t",
            index=False,
            quoting=3,
            doublequote=False,
            encoding="utf-8",
        )
