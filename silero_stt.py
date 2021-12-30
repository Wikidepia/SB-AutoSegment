import os

import ffmpeg
import torch
import torchaudio
import yt_dlp

ydl_opts = {
    "format": "bestaudio/best",
    "outtmpl": "%(id)s.%(ext)s",
    "quiet": True,
    "no_warnings": True,
}

model, decoder, _ = torch.hub.load(
    repo_or_dir="snakers4/silero-models",
    model="silero_stt",
    jit_model="jit_q",
)


def recognize(video_id):
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        metadata = ydl.extract_info(video_id, download=False)
        if metadata["duration"] > 15 * 60:
            raise Exception("Video is too long")
        ydl.download([video_id])

    ffmpeg.input(f"{video_id}.webm").output(
        f"{video_id}.wav", ac=1, ar=16000
    ).overwrite_output().run()

    audio, _ = torchaudio.load(f"{video_id}.wav")
    output = model(audio)
    wav_len = audio.shape[1] / 16000
    _, sentences = decoder(output[0].cpu(), wav_len, word_align=True)
    os.remove(f"{video_id}.wav")
    os.remove(f"{video_id}.webm")
    return sentences
