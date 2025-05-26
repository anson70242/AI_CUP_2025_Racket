from models.tcn import TemporalConvNet
from models.lstm import BasicLSTMClassifier
from models.mlp import MLP
from models.resnet50 import MyResNet50

import torch.nn as nn

# class TCN(nn.Module):
#     def __init__(self, time_step, num_channels, kernel_size, mlp_hidden_size,dropout, num_classes: list = [1, 1, 3, 4]):
#         super(TCN, self).__init__()
#         self.tcn = TemporalConvNet(
#             num_inputs=time_step, 
#             num_channels=num_channels, 
#             kernel_size=kernel_size, 
#             dropout=dropout
#         )
#         self.mlp = MLP(
#             input_dim=num_channels[-1],
#             num_classes=num_classes,
#             hidden_dim=mlp_hidden_size,
#             dropout=dropout
#         )

#     def forward(self, inputs):
#         features = self.tcn(inputs)
#         output = self.mlp(features[:, :, -1])
#         return output