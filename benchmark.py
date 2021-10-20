from __future__ import print_function
import argparse
import os
import time

import torch
import torch.backends.cudnn as cudnn
import numpy as np
import skimage
import torch.cuda as trc
from net_builder import SUPPORT_NETS, build_net, get_net_info, count_parameters 
import torch.nn.functional as F
import torch.nn as nn
import psutil
#from pytorch_memlab import MemReporter

process = psutil.Process(os.getpid())
cudnn.benchmark = True

def benchmark(model='resnet-101', batch_size=1, input_dim=[3, 32, 32]):

    net = build_net(model).cuda()

    input_shape = input_dim
    input_shape.insert(0, batch_size)
    get_net_info(net, input_shape=input_shape)
    print(input_shape)
    dummy_input = torch.randn(input_shape, dtype=torch.float).cuda()

    # INIT LOGGERS
    starter, ender = trc.Event(enable_timing=True), trc.Event(enable_timing=True)
    repetitions = 30
    mbytes = 2**20
    timings=np.zeros((repetitions,1))
    net.eval()
    #GPU-WARM-UP
    with torch.no_grad():
        for _ in range(10):
            _ = net(dummy_input)
            # MEASURE PERFORMANCE
    torch.cuda.empty_cache()
    time.sleep(1)
    with torch.no_grad():
        for rep in range(repetitions):
            starter.record()
            _ = net(dummy_input)
            ender.record()
            # WAIT FOR GPU SYNC
            trc.synchronize()
            curr_time = starter.elapsed_time(ender)
            timings[rep] = curr_time
            print(rep, curr_time)
    
    mean_syn = np.sum(timings) / repetitions
    std_syn = np.std(timings)
    print(model, mean_syn)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--net', type=str, help='indicate the name of net', default='', choices=SUPPORT_NETS)
    parser.add_argument('--batch-size', type=int, help='mini batch size', default=1)

    opt = parser.parse_args()
    benchmark(batch_size=8)
