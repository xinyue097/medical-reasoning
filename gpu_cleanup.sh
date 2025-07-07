#!/bin/bash

echo "=== GPU CLEANUP ==="
# Clear any existing GPU processes
echo "🧹 Clearing GPU processes and memory..."
python3 -c "
import torch
import gc
import subprocess
import sys
import os

# Clear GPU memory
if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        try:
            torch.cuda.set_device(i)
            torch.cuda.empty_cache()
        except Exception as e:
            print(f'  ❌ Failed to clear GPU {i}: {e}')

# Clear any DeepSpeed processes
try:
    subprocess.run(['pkill', '-f', 'deepspeed'], check=False, capture_output=True)
    subprocess.run(['pkill', '-f', 'torch.distributed'], check=False, capture_output=True)
except Exception as e:
    print(f'  ❌ Failed to clear processes: {e}')

gc.collect()
"