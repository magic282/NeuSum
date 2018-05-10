import math
import torch.optim as optim
import torch.nn as nn
from torch.nn.utils import clip_grad_norm
import neusum.modules

import logging

logger = logging.getLogger(__name__)


class Optim(object):
    def set_parameters(self, params):
        self.params = list(params)  # careful: params may be a generator
        if self.method == 'sgd':
            self.optimizer = optim.SGD(self.params, lr=self.lr)
        elif self.method == 'adagrad':
            self.optimizer = optim.Adagrad(self.params, lr=self.lr)
        elif self.method == 'adadelta':
            self.optimizer = optim.Adadelta(self.params, lr=self.lr)
        elif self.method == 'adam':
            # self.optimizer = optim.Adam(self.params, lr=self.lr)
            self.optimizer = neusum.modules.MyAdam(self.params, lr=self.lr)
        else:
            raise RuntimeError("Invalid optim method: " + self.method)

    def __init__(self, method, lr, max_grad_norm, max_weight_value=None, lr_decay=1, start_decay_at=None,
                 decay_bad_count=6):
        self.last_ppl = None
        self.lr = lr
        self.max_grad_norm = max_grad_norm
        self.max_weight_value = max_weight_value
        self.method = method
        self.lr_decay = lr_decay
        self.start_decay_at = start_decay_at
        self.start_decay = False
        self.decay_bad_count = decay_bad_count
        self.best_metric = 0
        self.bad_count = 0

    def step(self):
        # Compute gradients norm.
        if self.max_grad_norm:
            clip_grad_norm(self.params, self.max_grad_norm)
        self.optimizer.step()
        if self.max_weight_value:
            for p in self.params:
                p.data.clamp_(0 - self.max_weight_value, self.max_weight_value)

    # decay learning rate if val perf does not improve or we hit the start_decay_at limit
    def updateLearningRate(self, ppl, epoch):
        # if self.start_decay_at is not None and epoch >= self.start_decay_at:
        #     self.start_decay = True
        # if self.last_ppl is not None and ppl > self.last_ppl:
        #     self.start_decay = True
        #
        # if self.start_decay:
        #     self.lr = self.lr * self.lr_decay
        #     print("Decaying learning rate to %g" % self.lr)

        # self.last_ppl = ppl
        if ppl >= self.best_metric:
            self.best_metric = ppl
            self.bad_count = 0
        else:
            self.bad_count += 1
        logger.info('Bad_count: {0}\tCurrent lr: {1}'.format(self.bad_count, self.lr))
        logger.info('Best metric: {0}'.format(self.best_metric))

        if self.bad_count >= self.decay_bad_count and self.lr >= 1e-6:
            self.lr = self.lr * self.lr_decay
            logger.info("Decaying learning rate to %g" % self.lr)
            self.bad_count = 0
        self.optimizer.param_groups[0]['lr'] = self.lr
