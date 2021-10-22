from __future__ import print_function
from models import *
import copy
import torchvision.models as models

SUPPORT_NETS = {
        'resnet18': models.__dict__['resnet18'](),
        'resnet50': models.__dict__['resnet50'](),
        'resnet101': models.__dict__['resnet101'](),
        'densenet121': DenseNet121()
        }

INPUT_DIMS = {
        'resnet18':    [3, 32, 32],
        'resnet50':    [3, 224, 224],
        'resnet101':   [3, 224, 224],
        'densenet121': [3, 224, 224]
        }

def build_net(net_name):
    net  = SUPPORT_NETS.get(net_name, None)
    if net is None:
        print('Current supporting nets: %s , Unsupport net: %s', SUPPORT_NETS.keys(), net_name)
        raise 'Unsupport net: %s' % net_name
    return net

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def count_net_flops(net, data_shape=(1, 3, 224, 224)):
    from flops_counter import profile
    if isinstance(net, nn.DataParallel):
        net = net.module

    flop, _ = profile(copy.deepcopy(net), data_shape)
    return flop

def get_net_info(net, input_shape=(3, 224, 224), measure_latency=None, print_info=True):
    net_info = {}
    if isinstance(net, nn.DataParallel):
        net = net.module

    # parameters
    net_info['params'] = count_parameters(net) / 1e6

    # flops
    net_info['flops'] = count_net_flops(net, input_shape) / 1e6

    # latencies
    latency_types = [] if measure_latency is None else measure_latency.split('#')
    for l_type in latency_types:
        latency, measured_latency = measure_net_latency(net, l_type, fast=False, input_shape=input_shape)
        net_info['%s latency' % l_type] = {
            'val': latency,
            'hist': measured_latency
        }

    if print_info:
        print(net)
        print('Total training params: %.2fM' % (net_info['params']))
        print('Total FLOPs: %.2fM' % (net_info['flops']))
        for l_type in latency_types:
            print('Estimated %s latency: %.3fms' % (l_type, net_info['%s latency' % l_type]['val']))

    return net_info

