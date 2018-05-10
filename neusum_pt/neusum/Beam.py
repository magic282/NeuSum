from __future__ import division

# Class for managing the internals of the beam search process.
#
#
#         hyp1#-hyp1---hyp1 -hyp1
#                 \             /
#         hyp2 \-hyp2 /-hyp2#hyp2
#                               /      \
#         hyp3#-hyp3---hyp3 -hyp3
#         ========================
#
# Takes care of beams, back pointers, and scores.

import torch
import neusum

try:
    import ipdb
except ImportError:
    pass


class Beam(object):
    def __init__(self, size, cuda=False, force_max_len=False):

        self.size = size
        self.done = False
        self.force_max_len = force_max_len

        self.tt = torch.cuda if cuda else torch

        # The score for each translation on the beam.
        self.scores = self.tt.FloatTensor(size).zero_()
        self.all_scores = []
        self.all_length = []

        # The backpointers at each time-step.
        self.prevKs = []

        # The outputs at each time-step.
        self.nextYs = [self.tt.LongTensor(size).fill_(neusum.Constants.PAD)]
        self.nextYs[0][0] = neusum.Constants.BOS

        # The attentions (matrix) for each time.
        self.attn = []

    # Get the outputs for the current timestep.
    def getCurrentState(self):
        return self.nextYs[-1]

    # Get the backpointers for the current timestep.
    def getCurrentOrigin(self):
        return self.prevKs[-1]

    def set_illegal_position(self, pointer):
        """
        avoid duplicate selection
        :param pointer:
        :return:
        """
        hyp, attn = [], []
        for i in range(self.size):
            k = i
            for j in range(len(self.prevKs) - 1, -1, -1):
                h = self.nextYs[j + 1][k]
                pointer[i][h] = -float('inf')
                k = self.prevKs[j][k]
                # if self.force_max_len:
                #     pointer[i][s2s.Constants.END_OF_POINTER] = -float('inf')

    def get_doc_sent_mask(self, mask):
        res = mask.clone().unsqueeze(0).repeat(self.size, 1)
        for i in range(self.size):
            k = i
            for j in range(len(self.prevKs) - 1, -1, -1):
                h = self.nextYs[j + 1][k]
                res[i][h] = 1
                k = self.prevKs[j][k]
        return res

    # Given prob over words for every last beam `wordLk` and attention
    #   `attnOut`: Compute and update the beam search.
    #
    # Parameters:
    #
    #     * `wordLk`- probs of advancing from the last step (K x words)
    #     * `attnOut`- attention at the last step
    #
    # Returns: True if beam search is complete.
    def advance(self, pointer):
        numWords = pointer.size(1)

        # self.length += 1  # TODO: some is finished so do not acc length for them
        if len(self.prevKs) > 0:
            # finish_index = self.nextYs[-1].eq(s2s.Constants.END_OF_POINTER)
            # if any(finish_index):
            #     pointer.masked_fill_(finish_index.unsqueeze(1).expand_as(pointer), -float('inf'))
            #     for i in range(self.size):
            #         if self.nextYs[-1][i] == s2s.Constants.END_OF_POINTER:
            #             pointer[i][s2s.Constants.END_OF_POINTER] = 0
            # set up the current step length
            cur_length = self.all_length[-1]
            for i in range(self.size):
                # cur_length[i] += 0 if self.nextYs[-1][i] == s2s.Constants.END_OF_POINTER else 1
                cur_length[i] += 1
            self.set_illegal_position(pointer)

        # Sum the previous scores.
        if len(self.prevKs) > 0:
            prev_score = self.all_scores[-1]
            now_acc_score = pointer + prev_score.unsqueeze(1).expand_as(pointer)
            beamLk = now_acc_score / cur_length.unsqueeze(1).expand_as(now_acc_score)
        else:
            self.all_length.append(self.tt.FloatTensor(self.size).fill_(1))
            beamLk = pointer[0]

        flatBeamLk = beamLk.view(-1)

        bestScores, bestScoresId = flatBeamLk.topk(self.size, 0, True, True)
        self.scores = bestScores

        # bestScoresId is flattened beam x word array, so calculate which
        # word and beam each score came from
        prevK = bestScoresId / numWords
        predict = bestScoresId - prevK * numWords

        if len(self.prevKs) > 0:
            self.all_length.append(cur_length.index_select(0, prevK))
            self.all_scores.append(now_acc_score.view(-1).index_select(0, bestScoresId))
        else:
            self.all_scores.append(self.scores)

        self.prevKs.append(prevK)
        self.nextYs.append(predict)
        self.attn.append(pointer.index_select(0, prevK))

        # End condition is when every one is EOS.
        # if all(self.nextYs[-1].eq(s2s.Constants.END_OF_POINTER)):
        #     self.done = True

        return self.done

    def sortBest(self):
        return torch.sort(self.scores, 0, True)

    # Get the score of the best in the beam.
    def getBest(self):
        scores, ids = self.sortBest()
        return scores[1], ids[1]

    # Walk back to construct the full hypothesis.
    #
    # Parameters.
    #
    #     * `k` - the position in the beam to construct.
    #
    # Returns.
    #
    #     1. The hypothesis
    #     2. The attention at each time step.
    def getHyp(self, k):
        hyp, attn = [], []
        # print(len(self.prevKs), len(self.nextYs), len(self.attn))
        for j in range(len(self.prevKs) - 1, -1, -1):
            hyp.append(self.nextYs[j + 1][k])
            attn.append(self.attn[j][k])
            k = self.prevKs[j][k]

        return hyp[::-1], torch.stack(attn[::-1])
