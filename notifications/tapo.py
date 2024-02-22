from tapo import ApiClient
import time
import os

TAPO_USERNAME=os.environ['TAPO_USERNAME']
TAPO_PASSWORD=os.environ['TAPO_PASSWORD']
RIGHT_BULB=os.environ['RIGHT_BULB']
LEFT_BULB=os.environ['LEFT_BULB']

client = ApiClient(TAPO_USERNAME, TAPO_PASSWORD)

async def init():
  right_bulb = await client.l510(RIGHT_BULB)
  left_bulb = await client.l510(LEFT_BULB)

  await right_bulb.on()
  await left_bulb.on()

  return (right_bulb, left_bulb)

async def flicker():
  (right_bulb, left_bulb) = await init()

  for i in range(1, 8):
    right_brightness = 80 if i % 2 == 0 else 20
    right_bulb.set_brightness(right_brightness)
    left_brightness = 20 if i % 2 == 0 else 80
    left_bulb.set_brightness(left_brightness)
    time.sleep(1)

  print('resetting')
  right_bulb.set_brightness(20)
  left_bulb.set_brightness(20)

# For testing
# import asyncio
# asyncio.run(flicker())