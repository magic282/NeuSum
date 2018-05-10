import torch
import torch.nn as nn
import math


class MaxOut(nn.Module):
    def __init__(self, pool_size):
        super(MaxOut, self).__init__()
        self.pool_size = pool_size

    def forward(self, input):
        """
        input:
        reduce_size:
        """
        input_size = list(input.size())
        assert input_size[-1] % self.pool_size == 0
        output_size = [d for d in input_size]
        output_size[-1] = output_size[-1] // self.pool_size
        output_size.append(self.pool_size)
        last_dim = len(output_size) - 1
        input = input.view(*output_size)
        # TODO: This a temp fix
        if torch.__version__[:6] == '0.1.12':
            input, idx = input.max(last_dim)
        else:
            input, idx = input.max(last_dim, keepdim=True)
        output = input.squeeze(last_dim)

        return output

    def __repr__(self):
        return self.__class__.__name__ + '({0})'.format(self.pool_size)
