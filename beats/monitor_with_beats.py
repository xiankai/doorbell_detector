# Imports
print('starting up (imports, initialization, etc.)')
import torch
from torchaudio import transforms
from BEATs import BEATs, BEATsConfig
import sounddevice as sd
import numpy as np
import time
import requests
import os
from dotenv import load_dotenv
load_dotenv()

# Parameters
sampling_interval=float(os.environ['SAMPLING_INTERVAL'])
model_path=os.environ['MODEL_PATH']
recording_sample_rate=int(os.environ['RECORDING_SAMPLE_RATE'])
desired_sample_rate=16000 # specific to BEATS
buffer_size=15600
monitored_categories=','.split(os.environ['MONITORED_CATEGORIES'])
score_threshold=float(os.environ['SCORE_THRESHOLD'])
webhook_url=os.environ['WEBHOOK_URL']

# Initialization
checkpoint = torch.load(model_path)
cfg = BEATsConfig(checkpoint['cfg'])
BEATs_model = BEATs(cfg)
BEATs_model.load_state_dict(checkpoint['model'])
BEATs_model.eval()
audio_array = np.zeros((1, int(desired_sample_rate)))

# Define a function to resample audio
def resample(tensor):
  if recording_sample_rate != desired_sample_rate:
    resampler = transforms.Resample(orig_freq=recording_sample_rate, new_freq=desired_sample_rate)
    tensor = resampler(tensor)
  return tensor

# Inference loop
stream = sd.InputStream(device = 0, channels = 1, samplerate=recording_sample_rate, blocksize = buffer_size)
print('recording started')
with stream:
  while True:
    # Record audio from the stream
    audio_array, _ = stream.read(buffer_size)

    # Crude method of converting ndarray(15600,1) to Tensor.shape(3,10000) with copilot

    # Reshape the audio_array to have shape (3, 5200, 1)
    audio_array = audio_array.reshape(3, 5200, 1)
    # Pad the audio_array along the second dimension to have length 10000
    audio_array = np.pad(audio_array, ((0, 0), (0, 4800), (0, 0)), mode='constant')
    # Remove the last dimension of the audio_array
    audio_array = np.squeeze(audio_array)
    # Convert the audio_array to a PyTorch tensor
    tensor = torch.from_numpy(audio_array)

    padding_mask = torch.zeros(3, 10000).bool()
    # torch.no_grad is for performance https://github.com/microsoft/unilm/issues/998#issuecomment-1461310468
    with torch.no_grad():
      probs = BEATs_model.extract_features(tensor, padding_mask)[0]

    for i, (top5_label_prob, top5_label_idx) in enumerate(zip(*probs.topk(k=5))):
      top5_label = [checkpoint['label_dict'][label_idx.item()] for label_idx in top5_label_idx]
      print(f'Top 5 predicted labels of the {i}th audio are {top5_label} with probability of {top5_label_prob}')

    time.sleep(sampling_interval)