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
import pandas
from dotenv import load_dotenv
load_dotenv()

# Parameters
sampling_interval=float(os.environ['SAMPLING_INTERVAL'])
model_path=os.environ['MODEL_PATH']
recording_sample_rate=int(os.environ['RECORDING_SAMPLE_RATE'])
desired_sample_rate=16000 # specific to BEATS
buffer_size=10000
monitored_categories=','.split(os.environ['MONITORED_CATEGORIES'])
score_threshold=float(os.environ['SCORE_THRESHOLD'])
webhook_url=os.environ['WEBHOOK_URL']
labels_csv_file=os.environ['LABELS_CSV_FILE']

# Import the CSV file, conver to dict with 'mid' as key and 'display_name' as value
labels = pandas.read_csv(labels_csv_file).set_index('mid').to_dict()['display_name']

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

    # Crude method of converting ndarray(10000,1) to Tensor.shape(1,10000) with copilot
    tensor = torch.from_numpy(audio_array)
    tensor = torch.transpose(tensor, 0, 1)
    tensor = resample(tensor)

    padding_mask = torch.zeros(1, 10000).bool()
    # torch.no_grad is for performance https://github.com/microsoft/unilm/issues/998#issuecomment-1461310468
    with torch.no_grad():
      probs = BEATs_model.extract_features(tensor, padding_mask)[0]

    # for _, (top5_label_prob, top5_label_idx) in enumerate(zip(*probs.topk(k=5))):
    #   for i, label_idx  in enumerate(top5_label_idx):
    #     mid = checkpoint['label_dict'][label_idx.item()]
    #     label = labels[mid]
    #     prob = top5_label_prob[i]
    #     if prob > score_threshold:
    #       print(f'{label}: {prob}')
    # print('\n')

    for _, (top5_label_prob, top5_label_idx) in enumerate(zip(*probs.topk(k=5))):
      for i, label_idx  in enumerate(top5_label_idx):
        mid = checkpoint['label_dict'][label_idx.item()]
        label = labels[mid]
        prob = top5_label_prob[i]
        if prob > score_threshold and label in monitored_categories:
          try:
            response = requests.post(webhook_url, json={label: prob})
            if response.status_code == 200:
              time.sleep(8)
          except:
            pass

    time.sleep(sampling_interval)