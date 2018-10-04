from __future__ import division

import neusum
import torch
import argparse
import math
import time
import logging

logging.basicConfig(format='%(asctime)s [%(levelname)s:%(name)s]: %(message)s', level=logging.INFO)
file_handler = logging.FileHandler(time.strftime("%Y%m%d-%H%M%S") + '.log.txt', encoding='utf-8')
file_handler.setFormatter(logging.Formatter('%(asctime)s [%(levelname)-5.5s:%(name)s] %(message)s'))
logging.root.addHandler(file_handler)
logger = logging.getLogger(__name__)

parser = argparse.ArgumentParser(description='translate.py')

parser.add_argument('-model', required=True,
                    help='Path to model .pt file')
parser.add_argument('-src', required=True,
                    help='Source sequence to decode (one line per sequence)')
parser.add_argument('-tgt',
                    help='True target sequence (optional)')
parser.add_argument('-output', default='pred.txt',
                    help="""Path to output the predictions (each line will
                    be the decoded sequence""")
parser.add_argument('-beam_size', type=int, default=4,
                    help='Beam size')
parser.add_argument('-batch_size', type=int, default=64,
                    help='Batch size')
parser.add_argument('-max_sent_length', type=int, default=80,
                    help='Maximum sentence length.')
parser.add_argument('-replace_unk', action="store_true",
                    help="""Replace the generated UNK tokens with the source
                    token that had the highest attention weight. If phrase_table
                    is provided, it will lookup the identified source token and
                    give the corresponding target token. If it is not provided
                    (or the identified source token does not exist in the
                    table) then it will copy the source token""")
parser.add_argument('-verbose', action="store_true",
                    help='logger.info scores and predictions for each sentence')
parser.add_argument('-n_best', type=int, default=1,
                    help="""If verbose is set, will output the n_best
                    decoded sentences""")
parser.add_argument('-max_doc_len', type=int, default=80)
parser.add_argument('-max_decode_step', type=int, default=6)
parser.add_argument('-force_max_len', action="store_true")

parser.add_argument('-gpu', type=int, default=-1,
                    help="Device to run on")


def reportScore(name, scoreTotal, wordsTotal):
    logger.info("%s AVG SCORE: %.4f, %s PPL: %.4f" % (
        name, scoreTotal / wordsTotal,
        name, math.exp(-scoreTotal / wordsTotal)))


def addone(f):
    for line in f:
        yield line
    yield None


def addPair(f1, f2):
    for x, y1 in zip(f1, f2):
        yield (x, y1)
    yield (None, None)


def main():
    opt = parser.parse_args()
    seq_length = opt.max_sent_length
    logger.info(opt)
    opt.cuda = opt.gpu > -1
    if opt.cuda:
        torch.cuda.set_device(opt.gpu)

    translator = neusum.Summarizer(opt, logger=logger)

    outF = open(opt.output, 'w', encoding='utf-8')

    predScoreTotal, predWordsTotal, goldScoreTotal, goldWordsTotal = 0, 0, 0, 0

    srcBatch, tgtBatch = [], []
    src_raw, tgt_raw = [], []

    count = 0

    tgtF = open(opt.tgt) if opt.tgt else None
    for line in addone(open(opt.src, encoding='utf-8')):
        if line is not None:
            sline = line.strip()
            srcSents = sline.split('##SENT##')
            srcWords = [x.split(' ')[:seq_length] for x in srcSents]

            src_raw.append(srcSents)
            srcBatch.append(srcWords)

            if tgtF:
                tgtTokens = tgtF.readline().split(' ') if tgtF else None
                tgtBatch += [tgtTokens]
                # tgt_raw.append(tgtWords)

            if len(srcBatch) < opt.batch_size:
                continue
        else:
            # at the end of file, check last batch
            if len(srcBatch) == 0:
                break

        predBatch, predId, predScore, goldScore = translator.translate(srcBatch, src_raw, None)

        predScoreTotal += sum(score[0] for score in predScore)
        predWordsTotal += sum(len(x[0]) for x in predBatch)

        for b in range(len(predBatch)):
            count += 1
            outF.write('{0}\t{1}'.format(predId[b], predBatch[b]) + '\n')
            outF.flush()
        srcBatch, tgtBatch = [], []
        src_raw, tgt_raw = [], []

    if tgtF:
        tgtF.close()


if __name__ == "__main__":
    main()
