import os
from tflite_support.task import audio

desired_sample_rate=16000 # specific to yamnet
buffer_size=15600 # specific to yamnet
audio_format=audio.AudioFormat(1,desired_sample_rate)

classifier = None

def init(model_path):
  global classifier
  classifier = audio.AudioClassifier.create_from_file(os.path.expanduser(model_path))


def inference(tensor):
  classification_result = classifier.classify(tensor)

  return format(classification_result)

def format(classification_result):
  formatted = []
  for i in range(5):
    formatted.append((
      classification_result.categories[i].score,
      classification_result.categories[i].category_name,
    ))
  return formatted