import torch
import numpy as np
import torch.nn as nn
from torch.autograd import Variable,Function

import utils.frame_utils as frame_utils
import datasets
from datasets import StaticRandomCrop,StaticCenterCrop


try:
    from networks.resample2d_package.resample2d import Resample2d
    from networks.channelnorm_package.channelnorm import ChannelNorm
    from networks.correlation_package.correlation import Correlation
except:
    from .networks.resample2d_package.resample2d import Resample2d
    from .networks.channelnorm_package.channelnorm import ChannelNorm
    from networks.correlation_package.correlation import Correlation


def run_test(rgb_max = 255):
  
   device = torch.device('cuda')
   input_re_1 = Variable(torch.from_numpy(np.array(np.arange(0,1*2*3*4),np.float32)).resize(1,2,3,4).cuda(),requires_grad=True)
   input_re_2 = Variable(torch.from_numpy(np.array(np.arange(0,1*2*3*4),np.float32)).resize(1,2,3,4).cuda(),requires_grad=True)
   input_re_1 = input_re_1.cuda()
   input_re_2 = input_re_2.cuda()
   correlation_op = Correlation(pad_size=20, kernel_size=1, max_displacement=20, stride1=1, stride2=2,
                                  corr_multiply=1)
   output = correlation_op(input_re_1, input_re_2)
   print(output.sum())
#    ouputRe = output.resize(441*3*4)
#    print(ouputRe)
   output.backward(torch.ones(output.size()).cuda())
   print('result after bacward')
   print(input_re_1.grad.data)
   print(input_re_2.grad.data)
  
if __name__ == '__main__':
    run_test()
