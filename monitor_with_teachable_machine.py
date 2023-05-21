# Imports
print('starting up (imports, initialization, etc.)')
from tflite_support.task import audio
import time
import requests
import os
from dotenv import load_dotenv
load_dotenv()

# Parameters
sampling_interval=float(os.environ('SAMPLING_INTERVAL'))
model_path=os.environ('MODEL_PATH')
monitored_categories=','.split(os.environ('MONITORED_CATEGORIES'))
score_threshold=float(os.environ('SCORE_THRESHOLD'))
webhook_url=os.environ('WEBHOOK_URL')

# Initialization
classifier = audio.AudioClassifier.create_from_file(model_path)

  # Run inference
recorder = classifier.create_audio_record()
recorder.start_recording()
print('recording started')
while True:
  tensor_audio = classifier.create_input_tensor_audio()
  tensor_audio.load_from_audio_record(recorder)
  classification_result = classifier.classify(tensor_audio)
  classifications = classification_result.classifications[0]

  # current_time = time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime())
  # print(current_time)
  for category in classifications.categories:
    if category.category_name in monitored_categories and category.score > score_threshold:
      try:
        response = requests.get(webhook_url)
        if response.status_code == 200:
          time.sleep(8)
      except:
        pass

  #   print(category.category_name + ' ' + str(round(category.score, 2)))
  time.sleep(sampling_interval)
