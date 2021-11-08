from utils import PYTHON_HOME
import subprocess
import argparse
from net_builder import SUPPORT_NETS
import threading

class JobThread(threading.Thread):
    def __init__(self, process):
        super().__init__()
        self.process = process

    def run(self):
        self.process.communicate()

def batch_run(opt):

    cmd = []
    cmd.append(PYTHON_HOME)
    cmd.append('benchmark.py')
    cmd.append('--net')
    cmd.append(opt.net)
    cmd.append('--batch-size')
    cmd.append(str(opt.batch_size))
    cmd.append('--repetitions')
    cmd.append(str(opt.repetitions))
    cmd.append('--mode')
    cmd.append(opt.mode)
    cmd.append('--replicas')
    cmd.append(str(opt.replicas))
    print(cmd)

    p = subprocess.Popen(cmd)
    JobThread(p).start()
    return p

def dict_to_args(jd, opt):

    opt.net = jd['net']
    opt.batch_size = jd['batch_size']
    opt.repetitions = jd['repetitions']
    opt.mode = jd['mode']

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    opt = parser.parse_args()
    job1 = {'net': 'resnet50', 
            'batch_size': 32, 
            'repetitions': 400, 
            'mode': 'temporal'}

    job2 = {'net': 'resnet50', 
            'batch_size': 8, 
            'repetitions': 1600, 
            'mode': 'temporal'}

    dict_to_args(job1, opt)
    batch_run(opt)
    dict_to_args(job2, opt)
    batch_run(opt)

        
