import yaml
import argparse
import os

default_config_parser = parser = argparse.ArgumentParser(
    description='Training Config', add_help=False)

parser.add_argument(
    '-c',
    '--config',
    default='comsl.yaml',
    type=str,
    metavar='FILE',
    help='YAML config file specifying default arguments')

parser.add_argument(
    '--data_root',
    type=str,
    default=None,
    help="The root directory of CoVoST-2 audio clips")

parser.add_argument(
    '--cv_data_root',
    type=str,
    default=None,
    help="The root directory of Common Voice audio clips")

parser.add_argument(
    '--output_dir',
    type=str,
    default=None, )

parser.add_argument(
    '--ckpt_name',
    type=str,
    default="checkpoint-{epoch:02d}-{step}", )

parser.add_argument(
    '--num_nodes',
    type=int,
    default=1, )

parser.add_argument(
    '--language_list',
    type=str,
    nargs='+',
    default=None)

parser.add_argument(
    '--sample_rate',
    type=int,
    default=16000, )

parser.add_argument(
    '--valid_sample_rate',
    type=int,
    default=4, )


parser.add_argument(
    '--mode',
    type=str,
    default="resume", )

parser.add_argument(
    '--use_acti_ckpt',
    action='store_false', )

parser.add_argument(
    '--use_deepspeed',
    action='store_true',
)

parser.add_argument(
    '--ds_loss_scale',
    type=float,
    default=1.0, )

parser.add_argument(
    '--test_ckpt_name',
    type=str,
    default=None, )

parser.add_argument(
    '--chunk_size',
    type=int,
    default=11, )

# Optimizer and Scheduler
parser.add_argument(
    '--warmup_steps',
    type=int,
    default=5000, )

parser.add_argument(
    '--learning_rate',
    type=float,
    default=1e-5, )

parser.add_argument(
    '--adam_epsilon',
    type=float,
    default=1e-6, )

parser.add_argument(
    '--adam_betas',
    type=float,
    nargs='+',
    default=(0.9, 0.98), )

parser.add_argument(
    '--weight_decay',
    type=float,
    default=0.1, )

parser.add_argument(
    '--lr_pow',
    type=float,
    default=2.0, )

parser.add_argument(
    '--lr_end',
    type=float,
    default=1e-7, )

# For ComST
parser.add_argument(
    '--extra_language_list',
    type=str,
    nargs='+',
    default=None)

parser.add_argument(
    '--language_regularization_model_path',
    type=str,
    default=None,
)

parser.add_argument(
    '--language_init_model_path',
    type=str,
    default=None,
)

parser.add_argument(
    '--spch_init_model_path',
    type=str,
    default=None,
)

parser.add_argument(
    '--spch_n_layers',
    type=int,
    default=-1, )

parser.add_argument(
    '--erm_layer',
    type=int,
    default=4, )

parser.add_argument(
    '--p_mask',
    type=float,
    default=0.15, )

parser.add_argument(
    '--disable_spch_grad_epoch',
    type=int,
    default=0, )


def _parse_args_and_yaml(given_parser=None, config_path=None):
    if given_parser is None:
        given_parser = default_config_parser
    given_configs, remaining = given_parser.parse_known_args()
    file_name = given_configs.config if "yaml" in given_configs.config else given_configs.config + ".yaml"
    config_path = "config/exp_spec/" + file_name if config_path is None else config_path
    with open(config_path, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)
        given_parser.set_defaults(**cfg)

    args = given_parser.parse_args(remaining)

    args_text = yaml.safe_dump(args.__dict__, default_flow_style=False)
    return args, args_text


def parse_args_and_yaml(arg_parser=None, config_path=None):
    cfg = _parse_args_and_yaml(arg_parser, config_path)[0]

    if "OUTPUT_DIR" in os.environ:
        cfg.output_dir = os.environ["OUTPUT_DIR"]
    setattr(cfg, "log_output_dir", f"{cfg.output_dir}/logs")
    setattr(cfg, "check_output_dir", f"{cfg.output_dir}/ckpt")
    setattr(cfg, "cache_dir", f"{cfg.output_dir}/cache")
    if "DATA_ROOT" in os.environ:
        cfg.data_root = os.environ["DATA_ROOT"]
    if hasattr(cfg, "cv_data_root") and "CV_DATA_ROOT" in os.environ:
        cfg.cv_data_root = os.environ["CV_DATA_ROOT"]
    return cfg


if __name__ == "__main__":
    args, args_text = _parse_args_and_yaml()
    print(args_text)
    print(args.cache_dir)
