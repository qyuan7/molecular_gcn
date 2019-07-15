import torch
import torch.nn as nn


class gcn(nn.Module):

    def __init__(self,input_dim,hidden_dim,hidden_dim2):
        super().__init__()
        self.conv_linears = nn.ModuleList([nn.Linear(input_dim, hidden_dim)]
                                          +[nn.Linear(hidden_dim, hidden_dim) for i in range(2)])
        self.gates = nn.ModuleList([nn.Linear(hidden_dim, hidden_dim) for i in range(6)])
        self.readout = nn.Linear(hidden_dim, hidden_dim2)
        self.pred_layers = nn.ModuleList([nn.Linear(hidden_dim2, hidden_dim2) for i in range(2)])
        self.pred = nn.Linear(hidden_dim2, 1)


    def forward(self,input_X, input_A):

        def skip_connect(inpt_X, new_X):
            return inpt_X+new_X
        
        def skip_gate_connect(inpt_X, new_X, gate1, gate2):
            inpt_X = gate1(inpt_X)
            new_X = gate2(new_X)
            coeff = torch.sigmoid(inpt_X+new_X)
            out_X = torch.mul(coeff, new_X) + torch.mul((1-coeff), inpt_X)
            return out_X

        for i, l in enumerate(self.conv_linears):
            input_X = self.conv_linears[i](input_X)
            _input_X = torch.matmul(input_A, input_X)
            input_X = skip_gate_connect(input_X, _input_X, self.gates[2*i], self.gates[2*i+1])
            input_X = nn.functional.relu(input_X)
        input_X = self.readout(input_X)
        input_X = torch.sum(input_X, 1)
        input_X = torch.sigmoid(input_X)

        for j, l in enumerate(self.pred_layers):
            #print(i, l)
            input_X = self.pred_layers[j](input_X)
            if j == 0:
                input_X = torch.nn.functional.relu(input_X)
            elif j == 1:
                input_X = torch.tanh(input_X)
        input_X = self.pred(input_X)
        return input_X
