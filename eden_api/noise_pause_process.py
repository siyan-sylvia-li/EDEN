SAMPLING_RATE = 16000

import torch
torch.set_num_threads(1)

model, utils = torch.hub.load(repo_or_dir='snakers4/silero-vad',
                              model='silero_vad',
                              force_reload=False,
                              onnx=False)

(get_speech_timestamps,
 save_audio,
 read_audio,
 VADIterator,
 collect_chunks) = utils


def speech_only_audio():
    wav = read_audio("audio_cache/audio.wav", sampling_rate=SAMPLING_RATE)
    speech_timestamps = get_speech_timestamps(wav, model, sampling_rate=SAMPLING_RATE, return_seconds=True)
    pauses = []
    if len(speech_timestamps):
        for i, s in enumerate(speech_timestamps):
            if i == 0:
                continue
            pauses.append(speech_timestamps[i]["start"] - speech_timestamps[i - 1]["end"])
    if len(pauses) == 0:
        pauses = [0]
    return wav, sum(pauses) / len(pauses)

