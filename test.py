import torch
import numpy as np
import torch.nn as nn
from torch.autograd import Variable,Function

import utils.frame_utils as frame_utils
import datasets
from datasets import StaticRandomCrop,StaticCenterCrop


def run_test(rgb_max = 255):
  
    device = torch.device('cuda')
    input_re_1 = Variable(torch.from_numpy(np.array(np.arange(0,4*3*4*4),np.float32)).resize(4,3,4,4),requires_grad=True)
    input_re_2 = Variable(torch.from_numpy(np.array(np.arange(0,4*2*4*4),np.float32)).resize(4,2,4,4),requires_grad=True)
    input_re_1 = input_re_1.to(device)
    input_re_2 = input_re_2.to(device)
    resample_op = Resample2d()
    output = resample_op(input_re_1, input_re_2)
    output.backward(torch.ones(input_re_1.size()))
    print('result after bacward')
    print(input_re_1.grad.data)
    print(input_re_2.grad.data)
  
 if __name__ == '__main__':
    run_test()
