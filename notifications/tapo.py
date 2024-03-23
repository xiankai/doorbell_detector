from tapo import ApiClient
import time
import os

TAPO_USERNAME=os.environ['TAPO_USERNAME']
TAPO_PASSWORD=os.environ['TAPO_PASSWORD']
RIGHT_BULB=os.environ['RIGHT_BULB']
LEFT_BULB=os.environ['LEFT_BULB']

client = ApiClient(TAPO_USERNAME, TAPO_PASSWORD)

right_bulb, left_bulb = None, None

async def init():
  global right_bulb
  global left_bulb

  if not right_bulb:
    right_bulb = await client.l510(RIGHT_BULB)

  if not left_bulb:
    left_bulb = await client.l510(LEFT_BULB)

  return (right_bulb, left_bulb)

recently_flickered = False

async def flicker():
  global recently_flickered
  if recently_flickered:
    return
  else:
    recently_flickered = True

  was_off = False
  (right_bulb, left_bulb) = await init()

  device_info_json = await right_bulb.get_device_info_json()
  if device_info_json['device_on'] == False:
    was_off = True

  for i in range(1, 8):
    right_brightness = 80 if i % 2 == 0 else 20
    right_bulb.set_brightness(right_brightness)
    left_brightness = 20 if i % 2 == 0 else 80
    left_bulb.set_brightness(left_brightness)
    time.sleep(1)

  print('resetting')
  if was_off:
    await right_bulb.off()
    await left_bulb.off()
  else:
    right_bulb.set_brightness(20)
    left_bulb.set_brightness(20)

  time.sleep(10)
  recently_flickered = False

# For testing
# import asyncio
# asyncio.run(flicker())