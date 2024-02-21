import requests
import time
import os

webhook_url = os.environ['WEBHOOK_URL']

def ping(json):
  response = requests.post(webhook_url, json=json)
  if response.status_code == 200:
    time.sleep(8)