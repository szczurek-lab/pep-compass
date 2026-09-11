import pandas as pd
from collections import OrderedDict
import torch
from torch.autograd import Variable
from torch.nn import Linear, ReLU, ModuleList, CrossEntropyLoss, Sequential, Conv2d, MaxPool2d, Module, Softmax, BatchNorm2d, Dropout, L1Loss, Sigmoid, MSELoss, Tanh
from torch.optim import Adam, SGD
from tools_dataset import *
import tools
from tools import *
from tools.base import calRegMetrics, calRegMetrics
import numpy as np
import random
from sklearn.model_selection import KFold
def calulateConv2dOutsize(side_in=20, kernel_size=3, stride=1.0, padding=1, dilation=1):
    side_out = int((side_in + 2 * padding - dilation * (kernel_size - 1) -1) / stride + 1)
    return side_out

def calulate1ConvLayerOutNeurals(side_in=20, kernel_size=3, stride=1.0, padding=1, dilation=1):
    l1 = calulateConv2dOutsize(side_in=side_in, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation)
    mp1 = calulateConv2dOutsize(side_in=l1, kernel_size=2, stride=2, padding=0)
    return mp1

def calculateCNNoutSize(test_x, kernel_size=3, stride=1.0, padding=1, dilation=1, cnn_layers=2, cnn_filters=(10,100), cnn_low_spatial_concat=False):
    h_in, w_in = test_x.shape[2:]
    h_out, w_out = h_in, w_in
    cnn_out_size = h_in * w_in * cnn_filters[0]
    cnn_filters = cnn_filters[1:]
    print(f"CNN module input height & weight of each channel: {h_in}, {w_in}")
    for i in range(cnn_layers):
        w_out = calulate1ConvLayerOutNeurals(w_out, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation)
        h_out = calulate1ConvLayerOutNeurals(h_out, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation)
        cnn_out_size += w_out * h_out * cnn_filters[i]
        print(f"CNN module {i+1}-th layer output height & weight of each channel: {h_out}, {w_out}")
    if not cnn_low_spatial_concat:
        cnn_out_size = w_out * h_out * cnn_filters[i]
    print(f"CNN module output size of each channel: {cnn_out_size}")
    return cnn_out_size

class BaseNet(Module):
    def __init__(self, cnn_out_size=7500, kernel_size=3, stride=1, padding=1, dilation=1, groups=1, cnn_filters=(1, 10, 100),
                 cnn_low_spatial_concat=False, pred_label="EC_MIC", ldl_size=1000, del_refine=False, mat_scale=1,
                 act_fun='softmax', dropr=0.5, ldl_layers=(5000, 2500), out_layers=(100, 10), gpu_id=GPU_ID):
        super(BaseNet, self).__init__()
        self.cnn_low_spatial_concat, self.dropr = cnn_low_spatial_concat, dropr
        self.cnn_out_size, self.kernel_size, self.stride = cnn_out_size, kernel_size, stride
        self.padding, self.dilation, self.groups, self.cnn_filters = padding, dilation, groups, cnn_filters
        self.ldl_layers_tuple, self.out_layers_tuple = ldl_layers, out_layers

        device = torch.device("cuda:%d" % gpu_id if torch.cuda.is_available() else "cpu")
        print("device: ", device)
        mat = torch.arange(0, ldl_size, 1) * mat_scale
        mat = mat.to(device)
        self.ldl_size, self.del_refine, self.mat = ldl_size, del_refine, mat
        if act_fun == 'relu':
            self.act_fun = Tanh()
        if act_fun == "softmax":
            self.act_fun = Softmax(dim=1)
        if "p" in pred_label:
            self.act_fun = Tanh()
        else:
            self.act_fun = ReLU(inplace=True)
        self.cnn_layers, self.cnn_layers_list = self.CNNLayers(self.cnn_filters, drop_flag=True)
        self.ldl_layers = self.denseLayers(self.ldl_layers_tuple, in_num=cnn_out_size, out_num=self.ldl_size,
                                           act_fun=Softmax(dim=1), drop_flag=True)

        self.refine_layers = self.denseLayers(self.ldl_layers_tuple, in_num=cnn_out_size, out_num=self.ldl_size,
                                           act_fun=self.act_fun, drop_flag=True)

        self.out_layers = self.denseLayers(self.out_layers_tuple, in_num=self.ldl_size, out_num=1,
                                           act_fun=self.act_fun, drop_flag=False)
        print("cnn_out_size: ", cnn_out_size)

    def CNNLayers(self, cnn_filters, drop_flag):
        s = OrderedDict()
        cnn_list = []
        for i in range(len(cnn_filters)-1):
            s_temp = OrderedDict([
                ('Conv2d ' + str(i+1), Conv2d(in_channels=self.cnn_filters[i], out_channels=self.cnn_filters[i+1],
                                               kernel_size=self.kernel_size, stride=self.stride, padding=self.padding,
                                               dilation=self.dilation, groups=self.groups)),
                ('BatchNorm2d ' + str(i+1), BatchNorm2d(num_features=self.cnn_filters[i+1])),
                ('ReLU ' + str(i+1), ReLU(inplace=True)),
                ('MaxPool2d ' + str(i+1), MaxPool2d(kernel_size=2, stride=2)),
            ])
            s.update(s_temp)
            if drop_flag:
                s.update([('dropOut ' + str(i+1), Dropout(self.dropr))])
            cnn_list.append(Sequential(s_temp))
        s = Sequential(s)
        return s, cnn_list

    def denseLayers(self, layers_tuple, in_num, out_num, act_fun, drop_flag):
        layers_tuple = (in_num,) + layers_tuple + (out_num,)
        s = OrderedDict()
        for i in range(len(layers_tuple)-1):
            s_temp = OrderedDict([
                ('linear ' + str(i+1), Linear(layers_tuple[i], layers_tuple[i+1])),
                ('activation ' + str(i+1), act_fun)
            ])
            s.update(s_temp)
        if drop_flag:
            s.update([("dropOut", Dropout(self.dropr))])
        s = Sequential(s)
        return s

    def forward_cnnlayers(self, x):
        if self.cnn_low_spatial_concat:
            input_x = x.view(x.size(0), -1)
            # print("input shape: ", input_x.shape)
            output = []
            output.append(input_x)
            lx = x
            for i, l in enumerate(self.cnn_layers_list):
                # layerX = Sequential(l)
                lx = l(lx)
                lx_view = lx.view(lx.size(0), -1)
                # print("%d-th layer CNN output shape: " % (i+1), lx_view.shape)
                output.append(lx_view)
            output = torch.cat(output, dim=1)
            # print("normal CNN output shape: ", output.shape)
        else:
            layer2 = self.cnn_layers(x)
            # print("2nd normal CNN layer shape: ", layer2.shape)
            output = layer2.view(layer2.size(0), -1)
            # print("normal CNN output shape: ", output.shape)
        return output

class Net(BaseNet):
    def __init__(self, cnn_out_size=341, kernel_size=3, stride=1, padding=1, dilation=1, groups=1, cnn_filters=(10,100),
                 cnn_low_spatial_concat=False, pred_label="EC_MIC", ldl_size=1000, del_refine=False, mat_scale=1,
                 act_fun='softmax', dropr=0.5, ldl_layers=(5000, 2500), out_layers=(100, 10)):
        super(Net, self).__init__(cnn_out_size=cnn_out_size, kernel_size=kernel_size, stride=stride, padding=padding,
                                  dilation=dilation, groups=groups, cnn_filters=cnn_filters, del_refine=del_refine,
                                  cnn_low_spatial_concat=cnn_low_spatial_concat, pred_label=pred_label,
                                  ldl_size=ldl_size, mat_scale=mat_scale, act_fun=act_fun, dropr=dropr,
                                  ldl_layers=ldl_layers, out_layers=ldl_layers)
    # Defining the forward pass
    def forward(self, x):
        x = self.forward_cnnlayers(x)
        x1 = self.ldl_layers(x)
        if self.del_refine:
            crossMul = x1.mul(self.mat)
        else:
            x2 = self.refine_layers(x)
            crossMul = x1.mul(x2)
        out = self.out_layers(crossMul)
        return out, x1

class NetWithoutldl(BaseNet):
    def __init__(self, cnn_out_size=341, kernel_size=3, stride=1, padding=1, dilation=1, groups=1, cnn_filters=(10,100),
                 cnn_low_spatial_concat=False, pred_label="EC_MIC", ldl_size=1000, del_refine=False, mat_scale=1,
                 act_fun='softmax', dropr=0.5, ldl_layers=(5000, 2500), out_layers=(100, 10)):
        super(NetWithoutldl, self).__init__(cnn_out_size=cnn_out_size, kernel_size=kernel_size, stride=stride,
                                            padding=padding, dilation=dilation, groups=groups, cnn_filters=cnn_filters,
                                            cnn_low_spatial_concat=cnn_low_spatial_concat, pred_label=pred_label,
                                            ldl_size=ldl_size, del_refine=del_refine, mat_scale=mat_scale,
                                            act_fun=act_fun, dropr=dropr, ldl_layers=ldl_layers, out_layers=ldl_layers)

    # Defining the forward pass
    def forward(self, x):
        x = self.forward_cnnlayers(x)
        print("CNN out size: ", x.size())
        x = x.view(x.size(0), -1)
        if self.del_refine:
            x2 = x.mul(self.mat)
        else:
            x2 = self.refine_layers(x)
        out = self.out_layers(x2)
        return out


class NetWithoutldlAndCNN(BaseNet):
    def __init__(self, cnn_out_size=341, kernel_size=3, stride=1, padding=1, dilation=1, groups=1, cnn_filters=(10,100),
                 cnn_low_spatial_concat=False, pred_label="EC_MIC", ldl_size=1000, del_refine=False, mat_scale=1,
                 act_fun='softmax', dropr=0.5, ldl_layers=(5000, 2500), out_layers=(100, 10)):
        super(NetWithoutldlAndCNN, self).__init__(cnn_out_size=cnn_out_size, kernel_size=kernel_size, stride=stride,
                                            padding=padding, dilation=dilation, groups=groups, cnn_filters=cnn_filters,
                                            cnn_low_spatial_concat=cnn_low_spatial_concat, pred_label=pred_label,
                                            ldl_size=ldl_size, del_refine=del_refine, mat_scale=mat_scale,
                                            act_fun=act_fun, dropr=dropr, ldl_layers=ldl_layers, out_layers=ldl_layers)

        self.refine_layers = Sequential(
            Linear(cnn_out_size, 5000), # here is CNN input size
            self.act_fun,
            Linear(5000, self.ldl_size),
            self.act_fun,
        )

    # Defining the forward pass
    def forward(self, x):
        x = x.view(x.size(0), -1)
        x2 = self.refine_layers(x)
        out = self.out_layers(x2)
        return out