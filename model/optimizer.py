from transformers import get_polynomial_decay_schedule_with_warmup
import torch
try:
    from deepspeed.ops.adam import DeepSpeedCPUAdam
    deepspeed_available = True
except ImportError:
    deepspeed_available = False


def configure_optimizer_schedular(cfg, params_generator, num_training_steps, warmup_steps=None):
    no_decay = ["bias", "LayerNorm.weight", "layer_norm.weight", "attn_ln.weight", "mlp_ln"]
    warmup_steps = cfg.warmup_steps if warmup_steps is None else warmup_steps
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in params_generator()
                       if not any(nd in n for nd in no_decay)],
            "weight_decay": cfg.weight_decay,
        },
        {
            "params": [p for n, p in params_generator()
                       if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    if cfg.use_deepspeed and deepspeed_available:
        optimizer = DeepSpeedCPUAdam(optimizer_grouped_parameters,
                                     lr=cfg.learning_rate,
                                     eps=cfg.adam_epsilon,
                                     betas=cfg.adam_betas)
    else:
        optimizer = torch.optim.AdamW(optimizer_grouped_parameters,
                                      lr=cfg.learning_rate,
                                      eps=cfg.adam_epsilon,
                                      betas=cfg.adam_betas)

    scheduler = get_polynomial_decay_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=num_training_steps,
        power=cfg.lr_pow,
        lr_end=cfg.lr_end,
    )

    return optimizer, scheduler
