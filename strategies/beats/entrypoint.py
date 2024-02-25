import torch
from BEATs import BEATs, BEATsConfig
import os
from dotenv import load_dotenv
load_dotenv()

from pandas import read_csv
import inspect
current_file = inspect.getfile(inspect.currentframe())
import os
current_folder = os.path.dirname(current_file)
label_maps = read_csv(current_folder + '/class_labels_indices.csv').set_index('index')['display_name'].to_dict()

BEATs_model = None

def init(name):
  global BEATs_model
  checkpoint = torch.load(os.path.expanduser(name))
  cfg = BEATsConfig(checkpoint['cfg'])
  BEATs_model = BEATs(cfg)
  BEATs_model.load_state_dict(checkpoint['model'])
  BEATs_model.eval()

desired_sample_rate = 16000 # specific to BEATS

def inference(tensor):
  padding_mask = torch.zeros(1, 10000).bool()
  output = BEATs_model.extract_features(tensor, padding_mask)[0]

  return format(output)

def format(output):
  formatted = []
  for k, (prob, label) in enumerate(zip(*output.topk(5))):
    formatted.append((
      prob[k],
      label_maps[int(label[k])]
    ))

  return formatted
