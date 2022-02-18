import torch
from torch import nn
import torch.nn.functional as F

class LeNetFeatExtractor(nn.Module):
    def __init__(self):
        super(LeNetFeatExtractor, self).__init__()
        self.conv1 = nn.Conv2d(1, 128, 3)
        self.conv2 = nn.Conv2d(128, 16, 3)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        return x

class LeNetClassifier(nn.Module):
    def __init__(self):
        super(LeNetClassifier, self).__init__()
        self.fc1 = nn.Linear(16 * 6 * 6, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = torch.flatten(x,1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.feat = LeNetFeatExtractor()
        self.classifer = LeNetClassifier()

    def forward(self, x):
        x = self.feat(x)
        x = self.classifer(x)
        return x

import time
import numpy as np

import torch.backends.cudnn as cudnn
cudnn.benchmark = False

def benchmark(model, input_shape=(1024, 1, 32, 32), dtype='fp32', nwarmup=50, nruns=1000):
    input_data = torch.randn(input_shape)
    input_data = input_data.to("cuda")
    if dtype=='fp16':
        input_data = input_data.half()

    print("Warm up ...")
    with torch.no_grad():
        for _ in range(nwarmup):
            features = model(input_data)
    torch.cuda.synchronize()
    print("Start timing ...")
    timings = []
    with torch.no_grad():
        for i in range(1, nruns+1):
            start_time = time.time()
            features = model(input_data)
            torch.cuda.synchronize()
            end_time = time.time()
            timings.append(end_time - start_time)
            if i%100==0:
                print('Iteration %d/%d, ave batch time %.2f ms'%(i, nruns, np.mean(timings)*1000))

    print("Input shape:", input_data.size())
    print("Output features size:", features.size())

    print('Average batch time: %.2f ms'%(np.mean(timings)*1000))

model = LeNet()
model.to("cuda").eval()

#benchmark(model)



traced_model = torch.jit.trace(model, torch.empty([1,1,32,32]).to("cuda"))
traced_model

benchmark(traced_model)

model = LeNet().to("cuda").eval()

script_model = torch.jit.script(model)

script_model

benchmark(script_model)

import torch_tensorrt

# We use a batch-size of 1024, and half precision
compile_settings = {
    "inputs": [torch_tensorrt.Input(
            min_shape=[1024, 1, 32, 32],
            opt_shape=[1024, 1, 33, 33],
            max_shape=[1024, 1, 34, 34],
            dtype=torch.half
        )],
    "enabled_precisions": {torch.half} # Run with FP16
}

trt_ts_module = torch_tensorrt.compile(traced_model, **compile_settings)

input_data = torch.randn((1024, 1, 32, 32))
input_data = input_data.half().to("cuda")

input_data = input_data.half()
result = trt_ts_module(input_data)
torch.jit.save(trt_ts_module, "trt_ts_module.ts")


benchmark(trt_ts_module, input_shape=(1024, 1, 32, 32), dtype="fp16")


import torch_tensorrt

# We use a batch-size of 1024, and half precision
compile_settings = {
    "inputs": [torch_tensorrt.Input(
            min_shape=[1024, 1, 32, 32],
            opt_shape=[1024, 1, 33, 33],
            max_shape=[1024, 1, 34, 34],
            dtype=torch.half
        )],
    "enabled_precisions": {torch.half} # Run with FP16
}

trt_script_module = torch_tensorrt.compile(script_model, **compile_settings)

input_data = torch.randn((1024, 1, 32, 32))
input_data = input_data.half().to("cuda")

input_data = input_data.half()
result = trt_script_module(input_data)
torch.jit.save(trt_script_module, "trt_script_module.ts")


benchmark(trt_script_module, input_shape=(1024, 1, 32, 32), dtype="fp16")
