import torch
from torchaudio import transforms
from models.mn.model import get_model as get_mobilenet
from models.dymn.model import get_model as get_dymn
from models.preprocess import AugmentMelSTFT
from helpers.utils import NAME_TO_WIDTH, labels

import numpy as np

model = None
mel = None
desired_sample_rate=32000 # ref: https://github.com/fschmid56/EfficientAT/issues/17#issuecomment-1681842622

def init(name):
  # taken from default values in `inference.py`
  window_size=800
  hop_size=320
  n_mels=128
  strides=[2, 2, 2, 2]
  head_type="mlp"

  global model
  if name.startswith("dymn"):
    model = get_dymn(width_mult=NAME_TO_WIDTH(name),  pretrained_name=name, strides=strides)
  else:
      model = get_mobilenet(width_mult=NAME_TO_WIDTH(name), pretrained_name=name, strides=strides, head_type=head_type)
  model.eval()

  # model to preprocess waveform into mel spectrograms
  global mel
  mel = AugmentMelSTFT(n_mels=n_mels, sr=desired_sample_rate, win_length=window_size, hopsize=hop_size)
  mel.eval()

def inference(tensor):
  spec = mel(tensor)
  preds, features = model(spec.unsqueeze(0))
  return format(preds)

def format(preds):
  formatted = []
  preds = torch.sigmoid(preds.float()).squeeze().cpu().numpy()
  sorted_indexes = np.argsort(preds)[::-1]
  for k in range(5):
    label = labels[sorted_indexes[k]]
    prob = preds[sorted_indexes[k]]
    formatted.append((prob, label))

  return formatted