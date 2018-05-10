import torch
import torch.nn as nn
import math
import torch.nn.functional as F

try:
    import ipdb
except ImportError:
    pass


class ScoreAttention(nn.Module):
    def __init__(self, attend_dim, query_dim, att_dim):
        super(ScoreAttention, self).__init__()
        self.attend_dim = attend_dim
        self.query_dim = query_dim
        self.att_dim = att_dim
        self.linear_pre = nn.Linear(attend_dim, att_dim, bias=True)
        # self.linear_2 = nn.Linear(att_dim, att_dim, bias=True)
        self.linear_q = nn.Linear(query_dim, att_dim, bias=True)
        self.linear_v = nn.Linear(att_dim, 1, bias=True)
        if torch.__version__[:6] == '0.1.12':
            self.sm = nn.Softmax()
        else:
            self.sm = nn.Softmax(dim=1)
        self.mask = None

    def applyMask(self, mask):
        self.mask = mask

    def forward(self, input, context, precompute=None):
        """
        input: batch x dim
        context: batch x sourceL x dim
        """
        if precompute is None:
            precompute00 = self.linear_pre(context.contiguous().view(-1, context.size(2)))
            precompute = precompute00.view(context.size(0), context.size(1), -1)  # batch x sourceL x att_dim
        targetT = self.linear_q(input).unsqueeze(1)  # batch x 1 x att_dim

        tmp10 = precompute + targetT.expand_as(precompute)  # batch x sourceL x att_dim
        tmp20 = F.tanh(tmp10)
        energy = self.linear_v(tmp20.view(-1, tmp20.size(2))).view(tmp20.size(0), tmp20.size(1))  # batch x sourceL

        if self.mask is not None:
            energy = energy * (1 - self.mask) + self.mask * (-1e8)
        energy = F.softmax(energy, dim=1)

        return energy, precompute

    def __repr__(self):
        return self.__class__.__name__ + '(' + str(self.att_dim) + ' * ' + '(' \
               + str(self.attend_dim) + '->' + str(self.att_dim) + ' + ' \
               + str(self.query_dim) + '->' + str(self.att_dim) + ')' + ')'
