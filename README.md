# Introduction

This is the official repository of [ComSL: A Composite Speech-Language Model for End-to-End Speech-to-Text Translation](https://arxiv.org/abs/2305.14838), which includes the code for finetuning whisper and mbart, creating pseudo dataset and finetuning the ComSL model.

## Preparation

To run the code, first install the requirements:

```bash
pip install -r requirements.txt
```

Then, download CoVoST2 dataset following the instructions in [CoVoST2 Github Page](https://github.com/facebookresearch/covostz).

## Training

To launch the training, you should change the `data_root` in the config files in config/exp_spec to the root of CoVoST2 dataset. After that use command to start training:

```bash
python3 main.py -c XXX.yaml
```

where XXX.yaml is the configuration file in config/exp_spec.

## Training with Pseudo Data

In ouder to train with pesudo data, you should first download and extract Common Voice dataset from [Common Voice Website](https://commonvoice.mozilla.org/en/datasets). Then, modified the data path and pretrained model path in create_pseudo_data.py and run this script.

After that, set `cv_data_root` in config/exp_spec/comsl.yaml to the path of Common Voice dataset and uncomment the language in `avail_lang_extra`. Finally, run the training script as above.

```bash
python3 main.py -c comsl.yaml
```

## Citation

```bibtex
@misc{le2023comsl,
      title={ComSL: A Composite Speech-Language Model for End-to-End Speech-to-Text Translation}, 
      author={Chenyang Le and Yao Qian and Long Zhou and Shujie Liu and Michael Zeng and Xuedong Huang},
      year={2023},
      eprint={2305.14838},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```
