#!/bin/bash

echo "=== GPU TESTING ==="
echo "🔍 Testing GPU availability..."
python3 -c "
import torch
import os

# Check CUDA_VISIBLE_DEVICES
visible_devices = os.environ.get('CUDA_VISIBLE_DEVICES', 'Not set')
print(f'CUDA_VISIBLE_DEVICES: {visible_devices}')

if not torch.cuda.is_available():
    print('❌ CUDA not available')
    exit(1)

gpu_count = torch.cuda.device_count()
print(f'GPU Count: {gpu_count}')

if gpu_count != 2:
    print(f'❌ Expected 2 GPUs, found {gpu_count}')
    exit(1)

# Test each GPU
for i in range(gpu_count):
    try:
        torch.cuda.set_device(i)
        x = torch.randn(100, 100, device=f'cuda:{i}')
        y = torch.mm(x, x.t())
        print(f'GPU {i}: {torch.cuda.get_device_name(i)}')
        del x, y
        torch.cuda.empty_cache()
    except Exception as e:
        print(f'❌ GPU {i}: Test failed - {e}')
        exit(1)

print('All GPUs ready for training')
"

if [ $? -ne 0 ]; then
    echo "❌ GPU test failed, exiting"
    exit 1
fi