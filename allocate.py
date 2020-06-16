#!/usr/bin/env python

"""
Script to wait for an open GPU if needed, then to queue the job
if necessary. Use with `tsp` (task-spooler) for maximum synergy.
"""

import subprocess
import argparse
import time
import logging
import sys
import os, pwd
import nvgpu
from nvgpu.list_gpus import device_statuses

logging.basicConfig(level=logging.INFO)
mem_threshold = 90

def run(cmd):
    logging.info(cmd)
    subprocess.run([cmd], shell=True)

def _allocate_gpu(num_gpus):
    current_user = pwd.getpwuid(os.getuid()).pw_name
    gpu_info = nvgpu.gpu_info()
    device_info = device_statuses()

    # assume nothing is available
    completely_available = [False for _ in gpu_info]
    same_user_available = [False for _ in gpu_info]

    for i, (_info, _device) in enumerate(zip(gpu_info, device_info)):
        completely_available[i] = _device['is_available']
        unique_current_users = list(set(_device['users']))

        # if there's space on the gpu...
        if _info['mem_used_percent'] < mem_threshold:
            # ...and you're on this gpu...
            if current_user in unique_current_users:
                #...and you're the only one on this gpu...
                if len(unique_current_users) == 1:
                    # then allocate the gpu.
                    same_user_available[i] = True

    available_gpus = same_user_available
    if sum(same_user_available) == 0:
        available_gpus = completely_available

    available_gpus = [i for i, val in enumerate(available_gpus) if val]

    return available_gpus[:num_gpus]

if __name__ == "__main__":
    args = sys.argv

    num_gpus = int(sys.argv[1])
    cmd = sys.argv[2:]

    available_gpus = _allocate_gpu(num_gpus)

    while len(available_gpus) < num_gpus:
        logging.info("Waiting for available GPUs. Checking again in 30 seconds.")
        available_gpus = _allocate_gpu(num_gpus)
        time.sleep(30)

    available_gpus = ','.join(map(str, available_gpus))
    CUDA_VISIBLE_DEVICES = f'CUDA_VISIBLE_DEVICES={available_gpus}'
    cmd = ' '.join(cmd)
    cmd = f"{CUDA_VISIBLE_DEVICES} {cmd}"
    run(cmd)
