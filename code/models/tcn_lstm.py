import torch.nn as nn
from tcn import TemporalConvNet as TCN

class TCN_LSTM(nn.Module):
    def __init__(self):
        super(TCN_LSTM, self).__init__()
        self.tcn = TCN(num_inputs=7, channels=[32, 32, 32])
        self.lstm = nn.LSTM(input_size=32, hidden_size=64,
                            num_layers=2, batch_first=True)
        self.fc = nn.Linear(64, 1)

    def forward(self, x):
        x = x.permute(0, 2, 1)  # b i s
        x = self.tcn(x)  # b h s
        x = x.permute(0, 2, 1)  # b s h
        x, _ = self.lstm(x)  # b, s, h
        x = x[:, -1, :]
        x = self.fc(x)  # b output_size
        return x