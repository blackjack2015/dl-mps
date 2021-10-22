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
    print(cmd)

    p = subprocess.Popen(cmd)
    JobThread(p).start()
    return p

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--replicas', type=int, help='number of instances', default=1)
    parser.add_argument('--net', type=str, help='indicate the name of net', default='resnet18', choices=SUPPORT_NETS)
    parser.add_argument('--batch-size', type=int, help='mini batch size', default=1)
    parser.add_argument('--repetitions', type=int, help='iterations to run', default=100)
    parser.add_argument('--mode', type=str, help='temporal or mps', default='temporal')

    opt = parser.parse_args()
    for i in range(opt.replicas):
        batch_run(opt)
