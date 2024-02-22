import torch
import models

from pandas import read_csv
import inspect
current_file = inspect.getfile(inspect.currentframe())
import os
current_folder = os.path.dirname(current_file)
label_maps = read_csv(current_folder + '/class_labels_indices.csv').set_index('index')['display_name'].to_dict()

model = None
model_name = ""

def init(name):
  global model_name
  model_name = name
  global model
  model = getattr(models, model_name)(pretrained=True).eval()

desired_sample_rate = 16000 # ref: inference.py

def inference(tensor):
  zero_cache = None
  if 'SAT' in model_name:
    # First produce a "silence" cache
    *_, zero_cache = model(torch.zeros(1, int(model.cache_length / 100 * desired_sample_rate)), return_cache=True)

  if zero_cache is not None:
    output, zero_cache = model(tensor, cache=zero_cache, return_cache=True)
    output = output.squeeze(0)
  else:
    output = model(tensor).squeeze(0)

  return format(output)

def format(output):
  formatted = []
  for k, (prob, label) in enumerate(zip(*output.topk(5))):
    lab_idx = label.item()
    label_name = label_maps[lab_idx]
    formatted.append((prob, label_name))

  return formatted
