def get_module(name):
    if name == "mbart":
        from modules.mbart import MbartModelModule
        return MbartModelModule
    elif name == "whisper":
        from modules.whisper import WhisperModelModule
        return WhisperModelModule
    elif name == "whisper_asr":
        from modules.whisper_asr import WhisperAsrModelModule
        return WhisperAsrModelModule
    elif name == "cascade":
        from modules.cascade import CascadeModelModule
        return CascadeModelModule
    elif name == "comst":
        from modules.comsl import ComSTModule
        return ComSTModule
    else:
        raise NotImplementedError
