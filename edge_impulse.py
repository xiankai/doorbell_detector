import os
import sys, getopt
import signal
from edge_impulse_linux.audio import AudioImpulseRunner

from dotenv import load_dotenv
load_dotenv()
# Common labels
monitored_categories=os.environ['MONITORED_CATEGORIES'].split(',')
score_threshold=float(os.environ['SCORE_THRESHOLD'])
webhook_url=os.environ['WEBHOOK_URL']
model_name=os.environ['MODEL_NAME']
device_id=os.environ['DEVICE_ID']

import asyncio

notification_methods=os.environ['NOTIFICATION_METHODS'].split(',')
if 'webhook' in notification_methods:
  from notifications.webhook import ping
elif 'tapo' in notification_methods:
  from notifications.tapo import flicker

runner = None

def signal_handler(sig, frame):
  print('Interrupted')
  if (runner):
    runner.stop()
  sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)

with AudioImpulseRunner(model_name) as runner:
  try:
    model_info = runner.init()
    labels = model_info['model_parameters']['labels']
    print('Loaded runner for "' + model_info['project']['owner'] + ' / ' + model_info['project']['name'] + '"')

    for res, audio in runner.classifier(device_id=device_id):
      if 'stdout' in notification_methods:
        print('Result (%d ms.) ' % (res['timing']['dsp'] + res['timing']['classification']), end='')
      for label in labels:
        score = res['result']['classification'][label]
        if score > score_threshold and label in monitored_categories:
          if 'webhook' in notification_methods:
            asyncio.run(ping({[label]: score}))
          elif 'tapo':
            asyncio.run(flicker())
        if 'stdout' in notification_methods:
          print('%s: %.2f\t' % (label, score), end='')
      if 'stdout' in notification_methods:
        print('', flush=True)

  finally:
    if (runner):
      runner.stop()