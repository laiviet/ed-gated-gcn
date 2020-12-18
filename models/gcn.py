import torch
import torch.nn as nn
from layers.squeeze_embedding import SqueezeEmbedding
from layers.dynamic_rnn import DynamicLSTM

INFINITY_NUMBER = 1e12


class GraphConvolution(nn.Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, opt, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        self.nonlinearity = nn.Tanh()
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)

        # self.fc = nn.Linear(2 * 2 * opt.hidden_dim, opt.polarities_dim)
        # self.fc2 = nn.Linear(2 * 2 * opt.hidden_dim, opt.polarities_dim)
        # self.fc3 = nn.Linear(2 * 2 * opt.hidden_dim, opt.polarities_dim)
        # self.fc4 = nn.Linear(2 * 2 * opt.hidden_dim, opt.polarities_dim)

    def forward(self, text, adj):

        # print('GCN > text: ', tuple(text.shape))
        adj = adj.float()
        hidden = torch.matmul(text, self.weight)
        denom = torch.sum(adj, dim=2, keepdim=True) + 1

        # print('GCN > adj: ', tuple(adj.shape))
        # print('GCN > hidden: ', tuple(hidden.shape))
        # print('GCN > denom: ', tuple(denom.shape))

        output = torch.matmul(adj, hidden) / denom
        if self.bias is not None:
            return output + self.bias
        else:
            return output
