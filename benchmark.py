from __future__ import print_function
import argparse
import os
import time

import torch
import torch.backends.cudnn as cudnn
import numpy as np
import skimage
import torch.cuda as trc
from net_builder import SUPPORT_NETS, INPUT_DIMS, build_net, get_net_info, count_parameters 
from utils import logger, formatter
import torch.nn.functional as F
import torch.nn as nn
import psutil
import logging
#from pytorch_memlab import MemReporter

process = psutil.Process(os.getpid())
cudnn.benchmark = True
logRoot = 'logs/'

def benchmark(model='resnet-101', batch_size=1, repes=100):

    net = build_net(model).cuda()

    input_shape = INPUT_DIMS[model]
    input_shape.insert(0, batch_size)
    get_net_info(net, input_shape=input_shape)
    dummy_input = torch.randn(input_shape, dtype=torch.float).cuda()

    # INIT LOGGERS
    starter, ender = trc.Event(enable_timing=True), trc.Event(enable_timing=True)
    repetitions = repes * 3
    ep = repetitions // 3
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
            if rep % 50 == 0:
                print(rep, curr_time)
    
    mean_syn = np.sum(timings[ep:-ep]) / (repetitions-ep*2)
    std_syn = np.std(timings)
    logger.info('%s %f %f' % (model, mean_syn, std_syn))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--net', type=str, help='indicate the name of net', default='resnet18', choices=SUPPORT_NETS)
    parser.add_argument('--batch-size', type=int, help='mini batch size', default=1)
    parser.add_argument('--repetitions', type=int, help='iterations to run', default=100)
    parser.add_argument('--mode', type=str, help='temporal or mps', default='temporal')
    parser.add_argument('--replicas', type=int, help='number of instances', default=1)

    opt = parser.parse_args()
    logfile = '%s/%s/%s-bs%d-replicas%d' % (logRoot, opt.mode, opt.net, opt.batch_size, opt.replicas)
    hdlr = logging.FileHandler(logfile)
    hdlr.setFormatter(formatter)
    logger.addHandler(hdlr) 
    logger.info('Configurations: %s', opt)

    benchmark(model=opt.net, batch_size=opt.batch_size, repes=opt.repetitions)
