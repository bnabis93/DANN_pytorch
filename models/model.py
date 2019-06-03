import torch.nn as nn
from utils.functions import ReverseLayerF
import numpy as np
'''
nn.module documentation : https://pytorch.org/docs/stable/nn.html


'''

class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()
        
        '''
        nn.sequential
            A sequential container. Modules will be added to it in the order they are passed in the constructor. 
            Alternatively, an ordered dict of modules can also be passed in.
            To make it easier to understand, here is a small example:
                model = nn.Sequential(
                      nn.Conv2d(1,20,5),
                      nn.ReLU(),
                      nn.Conv2d(20,64,5),
                      nn.ReLU()
                    )
                sequential하게 쌓을 수 있는 container를 하나 준다고 보면 된다.

        Feature extractor
            feature extracting CNN model.
            end of feature extractor, you can show 2 paths (class classifier and doamin classifier)
            class classifier determine what is input's class (ex, person? chair?)
            domain classifier determine what is input's domain (source? target?)
            
        Class classifier
        
        Domain classifier
        
        '''
        self.feature = nn.Sequential()
        
        self.feature.add_module('f_conv1', nn.Conv2d(3,64, kernel_size = 5))
        self.feature.add_module('f_bn1',nn.BatchNorm2d(64))
        self.feature.add_module('f_pool1',nn.MaxPool2d(2))
        self.feature.add_module('f_relu1',nn.ReLU(True))
            
        self.feature.add_module('f_conv2', nn.Conv2d(64,50, kernel_size = 5))
        self.feature.add_module('f_bn2',nn.BatchNorm2d(50))
        self.feature.add_module('f_drop1',nn.Dropout2d())
        self.feature.add_module('f_pool2',nn.MaxPool2d(2))
        self.feature.add_module('f_relu2',nn.ReLU(True))
        
        #nn.Linear Applies a linear transformation to the incoming data: y = xA^T + by= xA T+b
        #nn.Linear(In, Out, Bias)
        self.class_classifier = nn.Sequential()
        self.class_classifier.add_module('c_fc1', nn.Linear(50 * 4 * 4, 100))
        self.class_classifier.add_module('c_bn1', nn.BatchNorm1d(100))
        self.class_classifier.add_module('c_relu1', nn.ReLU(True))
        self.class_classifier.add_module('c_drop1', nn.Dropout2d())
        self.class_classifier.add_module('c_fc2', nn.Linear(100, 100))
        self.class_classifier.add_module('c_bn2', nn.BatchNorm1d(100))
        self.class_classifier.add_module('c_relu2', nn.ReLU(True))
        self.class_classifier.add_module('c_fc3', nn.Linear(100, 10))
        self.class_classifier.add_module('c_softmax', nn.LogSoftmax(dim=1))
        
        # binary classifier, Source? Target?
        # dim (int) – A dimension along which LogSoftmax will be computed.
        # return 
        # a Tensor of the same dimension and shape as the input with values in the range [-inf, 0)
        
        self.domain_classifier = nn.Sequential()
        self.domain_classifier.add_module('d_fc1', nn.Linear(50 * 4 * 4, 100))
        self.domain_classifier.add_module('d_bn1', nn.BatchNorm1d(100))
        self.domain_classifier.add_module('d_relu1', nn.ReLU(True))
        self.domain_classifier.add_module('d_fc2', nn.Linear(100, 2))
        self.domain_classifier.add_module('d_softmax', nn.LogSoftmax(dim=1))
        
        
        
    def forward(self, input_data, alpha):
        '''
        what is view function ? : https://stackoverflow.com/questions/42479902/how-does-the-view-method-work-in-pytorch
        '''
        #mnist => 28 * 28
        #view = reshape
        
        input_data = input_data.expand(input_data.data.shape[0], 3, 28, 28)
        feature = self.feature(input_data)
        feature = feature.view(-1, 50 * 4 * 4)
        reverse_feature = ReverseLayerF.apply(feature, alpha)
        class_output = self.class_classifier(feature)
        domain_output = self.domain_classifier(reverse_feature)
        
        return class_output, domain_output