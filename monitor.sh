#! /bin/bash

nohup python3 monitor.py | ts '[%Y-%m-%d %H:%M:%S]' &