import torch
import torch.nn as nn
import torch.nn.functional as functional

def add_conv_stage(dim_in, dim_out, kernel_size=3, stride=1, padding=1, bias=True, useBN=False):
    if useBN:
        return nn.Sequential(
        nn.Conv2d(dim_in, dim_out, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias),
        nn.BatchNorm2d(dim_out),
        nn.Relu(),
        )
    else:
        return nn.Sequential(
        nn.Conv2d(dim_in, dim_out, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias),
        nn.ReLU(),
        )

def add_linear_block(dim_in, dim_out):
    return nn.Sequential(
        nn.Linear(dim_in, dim_out),
        nn.ReLU()
        )

class Net(nn.Module):
    def __init__(self, H, W, num_vertices, useBN=False):
        super(Net, self).__init__()

        self.conv1   = add_conv_stage(1, 16, useBN=useBN)
        self.conv2   = add_conv_stage(16, 16, useBN=useBN)
        self.conv3   = add_conv_stage(16, 16, useBN=useBN)

        self.linear1 = add_linear_block(30*16*25*37, 100)
        self.linear2 = add_linear_block(100, num_vertices*3)

        self.max_pool = nn.MaxPool2d(2)
        self.max_pool1 = nn.MaxPool1d(2)

    def forward(self, x):
        conv1_out = self.conv1(x)
        conv2_out = self.conv2(self.max_pool(conv1_out))
        
        conv3_out = self.conv3(self.max_pool(conv2_out))
        # shape: [30, 32, 200, 300] [ B, 32, H, W ]
        
        torch.Size([30, 32, 100, 150])
        torch.Size([30, 32, 50, 75])
        torch.Size([30, 32, 25, 37])
        
        conv3_out = self.max_pool(conv3_out)
        
        # torch.Size([30, 32, 25, 37])

        linear1_out = self.linear1(conv3_out.reshape(1, -1))
        linear1_out = nn.ReLU()(linear1_out)
        
        linear2_out = self.linear2(linear1_out)

        return linear2_out.reshape(-1,3)