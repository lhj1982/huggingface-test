# Use a pipeline as a high-level helper
from transformers import pipeline

pipe = pipeline("automatic-speech-recognition", model="openai/whisper-small")

result = pipe(
    "https://huggingface.co/datasets/Narsil/asr_dummy/resolve/main/mlk.flac")

print(result)
