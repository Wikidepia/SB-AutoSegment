import os

import ffmpeg
import torch
import yt_dlp

ydl_opts = {
    "format": "bestaudio/best",
    "outtmpl": "%(id)s.%(ext)s",
    "extractor_args": {
        "youtube": {
            "player_skip": ["webpage", "configs", "js"],
            "player_client": ["android"],
        }
    },
    "quiet": True,
    "no_warnings": True,
}

device = torch.device("cpu")
model, decoder, utils = torch.hub.load(
    repo_or_dir="snakers4/silero-models",
    model="silero_stt",
    language="en",
    device=device,
    jit_model="jit_q",
)
(read_batch, split_into_batches, read_audio, prepare_model_input) = utils


def recognize(video_id):
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        metadata = ydl.extract_info(video_id, download=False)
        if metadata["duration"] > 10 * 60:
            return []
        ydl.download([])
    ffmpeg.input(f"{video_id}.webm").output(
        f"{video_id}.wav", ac=1, ar=16000
    ).overwrite_output().run()

    inputs = prepare_model_input(read_batch([f"{video_id}.wav"]), device=device)
    wav_len = inputs.shape[1] / 16000
    output = model(inputs)
    _, sentences = decoder(output[0].cpu(), wav_len, word_align=True)
    os.remove(f"{video_id}.wav")
    os.remove(f"{video_id}.webm")
    return sentences
