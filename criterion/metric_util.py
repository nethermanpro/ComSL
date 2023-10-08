from sacrebleu.metrics.bleu import _get_tokenizer


def get_segment_tokenizers():
    return {
        "zh": _get_tokenizer("zh")(),
        "ja": _get_tokenizer("ja-mecab")(),
        "default": _get_tokenizer("13a")()
    }


def preprocess_sentence(s_list, lang, tokenizers):
    for i in range(len(s_list)):
        tokenizer = tokenizers.get(lang[i], tokenizers["default"])
        s_list[i] = "".join(tokenizer(s_list[i].rstrip()))
