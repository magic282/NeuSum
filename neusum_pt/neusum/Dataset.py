from __future__ import division

import math
import random

import torch
from torch.autograd import Variable

import neusum


class Dataset(object):
    def __init__(self, srcData, src_raw, tgtData, oracleData, src_rouge, batchSize, maxDocLen, cuda, volatile=False):
        self.src = srcData
        self.src_raw = src_raw
        if tgtData:
            self.tgt = tgtData
            assert (len(self.src) == len(self.tgt))
        else:
            self.tgt = None
        if oracleData:
            self.oracle = oracleData
            assert (len(self.src) == len(self.oracle))
        else:
            self.oracle = None
        if src_rouge:
            self.src_rouge = src_rouge
            assert (len(self.src) == len(self.src_rouge))
        else:
            self.src_rouge = None
        self.cuda = cuda

        self.batchSize = batchSize
        self.maxDocLen = maxDocLen
        self.numBatches = math.ceil(len(self.src) / batchSize)
        self.volatile = volatile

    def _batchify(self, data, align_right=False, include_lengths=False):
        lengths = [x.size(0) for x in data]
        max_length = max(lengths)
        out = data[0].new(len(data), max_length).fill_(neusum.Constants.PAD)
        for i in range(len(data)):
            data_length = data[i].size(0)
            offset = max_length - data_length if align_right else 0
            out[i].narrow(0, offset, data_length).copy_(data[i])

        if include_lengths:
            return out, lengths
        else:
            return out

    def __getitem__(self, index):
        assert index < self.numBatches, "%d > %d" % (index, self.numBatches)
        batch_src_data = self.src[index * self.batchSize:(index + 1) * self.batchSize]
        src_raw = self.src_raw[index * self.batchSize:(index + 1) * self.batchSize]
        doc_lengths = []
        buf = []
        for item in batch_src_data:
            doc_lengths.append(min(len(item), self.maxDocLen))
            buf += item[:self.maxDocLen]
            if len(item) < self.maxDocLen:
                buf += [torch.LongTensor([neusum.Constants.PAD]) for _ in range(self.maxDocLen - len(item))]

        srcBatch, lengths = self._batchify(buf, align_right=False, include_lengths=True)

        if self.tgt:
            tgtBatch = self.tgt[index * self.batchSize:(index + 1) * self.batchSize]
        else:
            tgtBatch = None
        if self.oracle:
            oracleBatch, oracleLength = self._batchify(
                self.oracle[index * self.batchSize:(index + 1) * self.batchSize],
                include_lengths=True)
        else:
            oracleBatch = None
        if self.src_rouge:
            buf = []
            max_points = max(oracleLength)
            batch_src_rouge_gain_data = self.src_rouge[index * self.batchSize:(index + 1) * self.batchSize]
            for item in batch_src_rouge_gain_data:
                buf += [x[:self.maxDocLen] for x in item]
                if len(item) < max_points:
                    buf += [torch.FloatTensor([neusum.Constants.PAD]) for _ in range(max_points - len(item))]
            src_rouge_batch = self._batchify(buf)
        else:
            src_rouge_batch = None

        # within batch sorting by decreasing length for variable length rnns
        indices = range(len(srcBatch))
        batch = zip(indices, srcBatch)
        batch, lengths = zip(*sorted(zip(batch, lengths), key=lambda x: -x[1]))
        indices, srcBatch = zip(*batch)

        def wrap(b):
            if b is None:
                return b
            b = torch.stack(b, 0).t().contiguous()
            if self.cuda:
                b = b.cuda()
            b = Variable(b, volatile=self.volatile)
            return b

        def simple_wrap(b):
            if b is None:
                return b
            if self.cuda:
                b = b.cuda()
            b = Variable(b, volatile=self.volatile)
            return b

        # wrap lengths in a Variable to properly split it in DataParallel
        lengths = torch.LongTensor(lengths).view(1, -1)
        lengths = Variable(lengths, volatile=self.volatile)

        doc_lengths = torch.LongTensor(doc_lengths).view(1, -1)
        doc_lengths = Variable(doc_lengths, volatile=self.volatile)

        if self.oracle:
            oracleLength = torch.LongTensor(oracleLength).view(1, -1)
            oracleLength = Variable(oracleLength, volatile=self.volatile)
        else:
            oracleLength = None

        return (wrap(srcBatch), lengths, doc_lengths), (src_raw,), \
               (tgtBatch,), (simple_wrap(oracleBatch), oracleLength), \
               simple_wrap(torch.LongTensor(list(indices)).view(-1)), (simple_wrap(src_rouge_batch),)

    def __len__(self):
        return self.numBatches

    def shuffle(self):
        data = list(zip(self.src, self.src_raw, self.tgt, self.oracle, self.src_rouge))
        self.src, self.src_raw, self.tgt, self.oracle, self.src_rouge = zip(
            *[data[i] for i in torch.randperm(len(data))])
