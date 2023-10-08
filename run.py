import logging
import os
from pathlib import Path

from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.strategies import DDPStrategy, DeepSpeedStrategy

from config.parse_yaml_args import parse_args_and_yaml
from data.data_util import load_data_record
from modules import get_module

os.environ["TORCH_DISTRIBUTED_DEBUG"] = "INFO"

cfg = parse_args_and_yaml()
module_name = cfg.module_name

Module = get_module(module_name)
seed_everything(42)

if __name__ == "__main__":

    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    joined_data_pair_lists, sep_data_pair_lists = {}, {}
    for split in ["train", "dev", "test"]:
        subsample_rate = cfg.valid_sample_rate if split == "dev" else 1
        language_list = cfg.language_list
        expanded_language_list = cfg.extra_language_list

        joined_data_pair_lists[split], \
        sep_data_pair_lists[split] = load_data_record(cfg.data_root,
                                                      split,
                                                      subsample_rate=subsample_rate,
                                                      language_list=language_list,
                                                      expanded_data_root=cfg.cv_data_root,
                                                      expanded_language_list=expanded_language_list, )

    Path(cfg.log_output_dir).mkdir(exist_ok=True)
    Path(cfg.check_output_dir).mkdir(exist_ok=True)
    Path(cfg.cache_dir).mkdir(exist_ok=True)

    tflogger = TensorBoardLogger(
        save_dir=cfg.log_output_dir,
        name=cfg.train_name,
        version=cfg.train_id
    )
    ckpt_dir = f"{cfg.check_output_dir}/checkpoint_{cfg.train_name}_{cfg.train_id}"

    monitor = cfg.monitor

    if "bleu" in monitor:
        mode = "max"
    elif "wer" in monitor or "loss" in monitor:
        mode = "min"
    else:
        raise NotImplementedError

    checkpoint_callback = ModelCheckpoint(
        monitor=monitor,
        mode=mode,
        dirpath=ckpt_dir,
        filename=cfg.ckpt_name,
        auto_insert_metric_name=False,
        save_top_k=5,  # all model save
        save_last=True,
    )

    callback_list = [checkpoint_callback, LearningRateMonitor(logging_interval="step")]

    model = Module(cfg, joined_data_pair_lists, sep_data_pair_lists)

    if cfg.use_deepspeed:
        strategy = DeepSpeedStrategy(
            stage=2,
            logging_level=logging.WARN,
            offload_optimizer=True,
            loss_scale=cfg.ds_loss_scale,
        )
    else:
        strategy = DDPStrategy(find_unused_parameters=False)

    trainer = Trainer(
        precision=16,
        num_nodes=cfg.num_nodes,
        accelerator="gpu",
        max_epochs=cfg.num_train_epochs,
        accumulate_grad_batches=cfg.gradient_accumulation_steps,
        logger=tflogger,
        callbacks=callback_list,
        strategy=strategy,
    )

    if cfg.test_ckpt_name is not None:
        trainer.test(model, ckpt_path=f"{ckpt_dir}/{cfg.test_ckpt_name}")
        exit()
    else:
        if cfg.mode == "test":
            trainer.test(model, ckpt_path="last")
        else:
            if cfg.mode == "resume":
                trainer.fit(model, ckpt_path="last")
            else:
                trainer.fit(model, ckpt_path=None)
            trainer.test(model, ckpt_path="best")
