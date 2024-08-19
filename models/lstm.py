
import torch
import torch.nn as nn 
import torch.functional as F


device = "cuda:0" if torch.cuda.is_available() else 'cpu'

input_dim = 3
hidden_dim = 100
sequence_dim = 28
layer_dim = 1
output_dim = 10



class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim):
        super(LSTMModel, self).__init__()
        # Hidden dimensions
        self.hidden_dim = hidden_dim

        # Number of hidden layers
        self.layer_dim = layer_dim

        self.lstm = nn.LSTM(input_dim, hidden_dim, layer_dim, batch_first=True)

        # Readout layer
        self.linear = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # Initialize hidden state with zeros
        x = x.view(x.size(0), 20, 3)
        h0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).to(device)
        # print(x.size())
        # Initialize cell state
        c0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).to(device)

        out, (hn, cn) = self.lstm(x, (h0.detach(), c0.detach()))

        out = self.linear(out[:, -1, :])
        # out.size() --> 100, 10
        return out



# input_dim = 3
# hidden_dim = 100
# sequence_dim = 28
# layer_dim = 1
# output_dim = 10
# model = LSTMModel(input_dim, hidden_dim, layer_dim, output_dim).to(device)
# print(model.eval)