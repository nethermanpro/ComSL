from Whisper.model import *
from torch.nn import LayerNorm
from transformers import MBartForConditionalGeneration, MBart50TokenizerFast, MBartConfig
import os.path as osp

MBART_PRETRAINED_MODEL = "facebook/mbart-large-50-many-to-many-mmt"


def load_mbart_tokenizer(cfg, extra_special_tokens=None):
    if extra_special_tokens is None:
        extra_special_tokens = []
    extra_special_tokens = ['cy_GB', 'ca_ES'] + extra_special_tokens

    tokenizer = MBart50TokenizerFast.from_pretrained(MBART_PRETRAINED_MODEL, cache_dir=cfg.cache_dir,
                                                     additional_special_tokens=extra_special_tokens)
    tokenizer.lang_code_to_id["cy_GB"] = tokenizer.convert_tokens_to_ids("cy_GB")
    tokenizer.lang_code_to_id["ca_ES"] = tokenizer.convert_tokens_to_ids("ca_ES")
    return tokenizer


def load_mbart_model(cfg, extra_special_tokens=None, load_from_local=True, path=None):
    if extra_special_tokens is None:
        extra_special_tokens = []

    configuration = MBartConfig.from_pretrained(MBART_PRETRAINED_MODEL, cache_dir=cfg.cache_dir)
    if hasattr(cfg, "attention_dropout"):
        configuration.attention_dropout = cfg.attention_dropout
    if hasattr(cfg, "dropout"):
        configuration.dropout = cfg.dropout
    mbart_model = MBartForConditionalGeneration.from_pretrained(MBART_PRETRAINED_MODEL, cache_dir=cfg.cache_dir,
                                                                config=configuration)
    mbart_model.resize_token_embeddings(configuration.vocab_size + 2 + len(extra_special_tokens))
    if path is None:
        path = cfg.language_init_model_path
    if load_from_local and path is not None:
        mbart_model.load_state_dict(torch.load(osp.join(cfg.cache_dir, path)))
        print("load mbart model from {}".format(osp.join(cfg.cache_dir, path)))

    return mbart_model


def lengths_to_padding_mask(lens):
    bsz, max_lens = lens.size(0), torch.max(lens).item()
    mask = torch.arange(max_lens).to(lens.device).view(1, max_lens)
    mask = mask.expand(bsz, -1) >= lens.view(bsz, 1).expand(-1, max_lens)
    return mask


class Conv1dAdaptor(nn.Module):
    def __init__(
            self,
            in_dim,
            out_dim,
            n_layers=3,
            kernel_size=3,
            stride=2,
            layerdrop=0.0,
            layernorm=False,
            proj=False,
    ):
        super().__init__()
        self.proj, self.proj_ln = None, None
        self.post_proj, self.post_proj_ln = None, None
        if proj:
            self.proj = nn.Sequential(
                nn.Linear(in_dim, in_dim * 4), nn.ReLU(), nn.Linear(in_dim * 4, in_dim)
            )
            self.proj_ln = LayerNorm(in_dim)
            self.post_proj = nn.Sequential(
                nn.Linear(out_dim, out_dim * 4),
                nn.ReLU(),
                nn.Linear(out_dim * 4, out_dim),
            )
            self.post_proj_ln = LayerNorm(out_dim)

        self.layers = nn.ModuleList(
            nn.Conv1d(
                in_dim if i == 0 else out_dim,
                out_dim * 2,
                kernel_size,
                stride=stride,
                padding=kernel_size // 2,
            )
            for i in range(n_layers)
        )
        self.stride = stride
        self.layerdrop = layerdrop
        self.layernorm = LayerNorm(in_dim) if layernorm else None

    def forward(self, x, padding_mask: Optional[torch.Tensor] = None):
        if self.layernorm is not None:
            x = self.layernorm(x)

        if self.proj is not None:
            x = x + 0.5 * self.proj(x)
            x = self.proj_ln(x)

        # B x T x C -> B x C x T
        x = x.transpose(1, 2)
        out_lens = None
        if padding_mask is not None:
            out_lens = (~padding_mask).sum(1).float()

        for layer in self.layers:
            layerdrop_prob = np.random.random()
            if not self.training or (layerdrop_prob > self.layerdrop):
                x = nn.functional.glu(layer(x), dim=1)
                if padding_mask is not None:
                    out_lens = ((out_lens - 1) / self.stride + 1).floor()
        # B x C x T -> B x T x C
        x = x.transpose(1, 2)

        if self.post_proj is not None:
            x = x + 0.5 * self.post_proj(x)
            x = self.post_proj_ln(x)

        out_padding_mask = None
        if padding_mask is not None:
            out_padding_mask = lengths_to_padding_mask(out_lens.long())
        return x, out_padding_mask


class GradMultiply(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, scale):
        ctx.scale = scale
        res = x.new(x)
        return res

    @staticmethod
    def backward(ctx, grad):
        return grad * ctx.scale, None


def deep_to_device(obj, device):
    if isinstance(obj, torch.Tensor):
        return obj.to(device)
    elif isinstance(obj, dict):
        return {k: deep_to_device(v, device) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [deep_to_device(v, device) for v in obj]
    else:
        return obj
