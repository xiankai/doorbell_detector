# Imports
print('starting up (imports, initialization, etc.)')
from tflite_support.task import audio
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
desired_sample_rate=16000 # specific to yamnet
buffer_size=15600 # specific to yamnet
monitored_categories=','.split(os.environ['MONITORED_CATEGORIES'])
score_threshold=float(os.environ['SCORE_THRESHOLD'])
webhook_url=os.environ['WEBHOOK_URL']

# Initialization
classifier = audio.AudioClassifier.create_from_file(model_path)
audio_array = np.zeros((int(desired_sample_rate), 1))

def audio_callback(indata, frames, time, status):
  # Downsample the input data
  global audio_array
  audio_array = indata[::int(recording_sample_rate/desired_sample_rate)]

def format_categories(categories):
  formatted_categories = {}
  for category in categories:
    formatted_categories[category.category_name] = round(category.score, 2)
  return formatted_categories

# Inference loop
stream = sd.InputStream(device = 1, channels = 1, samplerate=recording_sample_rate, callback = audio_callback, blocksize = buffer_size)
audio_format=audio.AudioFormat(1,desired_sample_rate)
print('recording started')
with stream:
  while True:
    tensor_audio = audio.TensorAudio(audio_format=audio_format,buffer_size=buffer_size)
    tensor_audio.load_from_array(src=audio_array)
    classification_result = classifier.classify(tensor_audio)
    classifications = classification_result.classifications[0]
    categories = classifications.categories

    # current_time = time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime())
    # print("\n")
    # print(current_time)
    # json_data = {}
    # json_data['categories'] = format_categories(categories[:5])
    # time.sleep(1000)
    if categories[0].category_name in monitored_categories and categories[0].score > score_threshold:
      try:
        response = requests.post(webhook_url, json=format_categories(categories[:5]))
        if response.status_code == 200:
          time.sleep(8)
      except:
        pass
    time.sleep(sampling_interval)