from __future__ import division

import os
import math
import time
import logging
import argparse
import torch
import torch.nn as nn
from torch import cuda
from torch.autograd import Variable
import torch.nn.functional as F

try:
    import ipdb
except ImportError:
    pass
import neusum
from neusum.xinit import xavier_normal, xavier_uniform
import xargs
from PyRouge.Rouge import Rouge

parser = argparse.ArgumentParser(description='train.py')
xargs.add_data_options(parser)
xargs.add_model_options(parser)
xargs.add_train_options(parser)

opt = parser.parse_args()

logging.basicConfig(format='%(asctime)s [%(levelname)s:%(name)s]: %(message)s', level=logging.INFO)
log_file_name = time.strftime("%Y%m%d-%H%M%S") + '.log.txt'
if opt.log_home:
    log_file_name = os.path.join(opt.log_home, log_file_name)
file_handler = logging.FileHandler(log_file_name, encoding='utf-8')
file_handler.setFormatter(logging.Formatter('%(asctime)s [%(levelname)-5.5s:%(name)s] %(message)s'))
logging.root.addHandler(file_handler)
logger = logging.getLogger(__name__)

logger.info('My PID is {0}'.format(os.getpid()))
logger.info(opt)

if torch.cuda.is_available() and not opt.gpus:
    logger.info("WARNING: You have a CUDA device, so you should probably run with -gpus 0")

if opt.seed > 0:
    torch.manual_seed(opt.seed)

if opt.gpus:
    if opt.cuda_seed > 0:
        torch.cuda.manual_seed(opt.cuda_seed)
    cuda.set_device(opt.gpus[0])

logger.info('My seed is {0}'.format(torch.initial_seed()))
logger.info('My cuda seed is {0}'.format(torch.cuda.initial_seed()))


def regression_loss(pred_scores, gold_scores, mask, crit):
    """

    :param pred_scores: (step, batch, doc_len)
    :param gold_scores: (batch*step, doc_len)
    :param mask: (batch, doc_len)
    :param crit:
    :return:
    """
    pred_scores = pred_scores.transpose(0, 1).contiguous()  # (batch, step, doc_len)
    if isinstance(crit, nn.KLDivLoss):
        # TODO: we better use log_softmax(), not log() here. log_softmax() is more numerical stable.
        pred_scores = torch.log(pred_scores + 1e-8)
    gold_scores = gold_scores.view(*pred_scores.size())
    loss = crit(pred_scores, gold_scores)
    loss = loss * (1 - mask).unsqueeze(1).expand_as(loss)
    reduce_loss = loss.sum()
    return reduce_loss


def addPair(f1, f2):
    for x, y1 in zip(f1, f2):
        yield (x, y1)
    yield (None, None)


def load_dev_data(summarizer, src_file, tgt_file):
    seq_length = opt.max_sent_length
    dataset, raw = [], []
    src_raw, tgt_raw = [], []
    src_batch, tgt_batch = [], []
    srcF = open(src_file, encoding='utf-8')
    tgtF = open(tgt_file, encoding='utf-8')

    for sline, tline in addPair(srcF, tgtF):
        if (sline is not None) and (tline is not None):
            if sline == "" or tline == "":
                continue
            sline = sline.strip()
            tline = tline.strip()
            srcSents = sline.split('##SENT##')
            tgtSents = tline.split('##SENT##')
            srcWords = [x.split(' ')[:seq_length] for x in srcSents]
            tgtWords = ' '.join(tgtSents)
            src_raw.append(srcSents)
            src_batch.append(srcWords)
            tgt_raw.append(tgtWords)

            if len(src_batch) < opt.batch_size:
                continue
        else:
            # at the end of file, check last batch
            if len(src_batch) == 0:
                break
        data = summarizer.buildData(src_batch, src_raw, tgt_raw, None, None)
        dataset.append(data)
        src_batch, tgt_batch = [], []
        src_raw, tgt_raw = [], []
    srcF.close()
    tgtF.close()
    return dataset


evalModelCount = 0
totalBatchCount = 0
rouge_calculator = Rouge.Rouge(use_ngram_buf=True)


def evalModel(model, summarizer, evalData):
    global evalModelCount
    global rouge_calculator
    evalModelCount += 1
    predict, gold = [], []
    predict_id = []
    dataset = evalData
    for data in dataset:
        """
        input: (wrap(srcBatch), lengths, doc_lengths), (src_raw,), \
               (tgtBatch,), (oracleBatch, oracleLength), \
               Variable(torch.LongTensor(list(indices)).view(-1).cuda(), volatile=self.volatile)
        """
        #  (2) translate
        batch = data[0]
        pred, predScore, attn, _ = summarizer.translateBatch(batch)
        # pred, predScore, attn = list(zip(
        #     *sorted(zip(pred, predScore, attn, indices),
        #             key=lambda x: x[-1])))[:-1]

        #  (3) convert indexes to words
        predBatch = []
        src_raw = batch[1][0]
        for b in range(len(src_raw)):
            n = 0
            selected_sents = []
            selected_id = []
            for idx in pred[b][n]:
                # if idx == 0 or idx - 1 >= len(src_raw[b]):
                #     break
                # selected_sents.append(src_raw[b][idx - 1])
                # selected_id.append(idx - 1)

                # no sentinel
                if idx >= len(src_raw[b]):
                    break
                selected_sents.append(src_raw[b][idx])
                selected_id.append(idx)
            predBatch.append(' '.join(selected_sents))
            predict_id.append(tuple(selected_id))
        tgt_raw = batch[2][0]
        gold += tgt_raw
        predict += predBatch
    scores = rouge_calculator.compute_rouge(gold, predict)
    with open('dev.out.{0}'.format(evalModelCount), 'w', encoding='utf-8') as of:
        for p, idx in zip(predict, predict_id):
            of.write('{0}\t{1}'.format(idx, p) + '\n')
    return scores['rouge-2']['f']


def gen_pointer_mask(target_lengths, batch_size, max_step):
    res = torch.ByteTensor(batch_size, max_step).fill_(1)
    for idx, ll in enumerate(target_lengths.data[0]):
        if ll == max_step:
            continue
        res[idx][ll:] = 0
    res = res.float()
    return res


def trainModel(model, summarizer, trainData, validData, dataset, optim):
    logger.info(model)
    model.train()
    logger.warning("Set model to {0} mode".format('train' if model.training else 'eval'))

    # define criterion of each GPU
    regression_crit = nn.KLDivLoss(size_average=False, reduce=False)

    def saveModel(metric=None):
        model_state_dict = model.module.state_dict() if len(opt.gpus) > 1 else model.state_dict()
        model_state_dict = {k: v for k, v in model_state_dict.items() if 'generator' not in k}
        #  (4) drop a checkpoint
        checkpoint = {
            'model': model_state_dict,
            'dicts': dataset['dicts'],
            'opt': opt,
            'epoch': epoch,
            'optim': optim
        }
        save_model_path = 'model'
        if opt.save_path:
            if not os.path.exists(opt.save_path):
                os.makedirs(opt.save_path)
            save_model_path = opt.save_path + os.path.sep + save_model_path
        if metric is not None:
            torch.save(checkpoint, '{0}_devRouge_{1}_e{2}.pt'.format(save_model_path, round(metric, 4), epoch))
        else:
            torch.save(checkpoint, '{0}_e{1}.pt'.format(save_model_path, epoch))

    def trainEpoch(epoch):

        if opt.extra_shuffle and epoch > opt.curriculum:
            logger.info('Shuffling...')
            trainData.shuffle()

        # shuffle mini batch order
        batchOrder = torch.randperm(len(trainData))

        total_reg_loss = total_point_loss = total_docs = total_points = 0
        start = time.time()
        for i in range(len(trainData)):
            global totalBatchCount
            totalBatchCount += 1
            """
            (wrap(srcBatch), lengths, doc_lengths), (src_raw,), \
               (tgtBatch,), (simple_wrap(oracleBatch), oracleLength), \
               simple_wrap(torch.LongTensor(list(indices)).view(-1)), (simple_wrap(src_rouge_batch),)
            """
            batchIdx = batchOrder[i] if epoch > opt.curriculum else i
            batch = trainData[batchIdx]

            model.zero_grad()
            doc_sent_scores, doc_sent_mask = model(batch)  # (step, batch, doc_len), (batch, doc_len)

            # regression loss
            gold_sent_rouge_scores = batch[5][0]  # (batch*step, doc_len)
            reg_loss = regression_loss(doc_sent_scores, gold_sent_rouge_scores, doc_sent_mask, regression_crit)

            loss = reg_loss

            report_reg_loss = reg_loss.data[0]
            report_point_loss = 0
            num_of_docs = doc_sent_mask.size(0)
            num_of_pointers = 0
            total_reg_loss += report_reg_loss
            total_point_loss += report_point_loss
            total_docs += num_of_docs
            total_points += num_of_pointers

            # update the parameters
            loss.backward()
            optim.step()

            if i % opt.log_interval == -1 % opt.log_interval:
                logger.info(
                    "Epoch %2d, %6d/%5d/%5d; reg_loss: %6.2f; docs: %5d; avg_reg_loss: %6.2f; %6.0f s elapsed" %
                    (epoch, totalBatchCount, i + 1, len(trainData),
                     report_reg_loss,
                     num_of_docs,
                     report_reg_loss / num_of_docs,
                     time.time() - start))

                start = time.time()

            if validData is not None and totalBatchCount % opt.eval_per_batch == -1 % opt.eval_per_batch \
                    and totalBatchCount >= opt.start_eval_batch:
                model.eval()
                logger.warning("Set model to {0} mode".format('train' if model.training else 'eval'))
                valid_bleu = evalModel(model, summarizer, validData)
                model.train()
                logger.warning("Set model to {0} mode".format('train' if model.training else 'eval'))
                logger.info(valid_bleu)
                valid_bleu = valid_bleu[0]
                logger.info('Validation Score: %g' % (valid_bleu * 100))
                if valid_bleu >= optim.best_metric:
                    saveModel(valid_bleu)
                optim.updateLearningRate(valid_bleu, epoch)

        return total_reg_loss / total_docs

    for epoch in range(opt.start_epoch, opt.epochs + 1):
        logger.info('')
        #  (1) train for one epoch on the training set
        train_reg_loss = trainEpoch(epoch)
        logger.info('Train regression loss: %g' % train_reg_loss)
        if opt.dump_epoch_checkpoint:
            logger.info('Saving checkpoint for epoch {0}...'.format(epoch))
            saveModel()


def main():
    if not opt.online_process_data:
        raise Exception('This code does not use preprocessed .pt pickle file. It has some issues with big files.')
        # dataset = torch.load(opt.data)
    else:
        import onlinePreprocess
        onlinePreprocess.seq_length = opt.max_sent_length
        onlinePreprocess.max_doc_len = opt.max_doc_len
        onlinePreprocess.shuffle = 1 if opt.process_shuffle else 0
        onlinePreprocess.norm_lambda = opt.norm_lambda
        from onlinePreprocess import prepare_data_online
        dataset = prepare_data_online(opt.train_src, opt.src_vocab, opt.train_tgt, opt.tgt_vocab, opt.train_oracle,
                                      opt.train_src_rouge)

    trainData = neusum.Dataset(dataset['train']['src'], dataset['train']['src_raw'],
                            dataset['train']['tgt'], dataset['train']['oracle'], dataset['train']['src_rouge'],
                            opt.batch_size, opt.max_doc_len, opt.gpus)
    dicts = dataset['dicts']
    logger.info(' * vocabulary size. source = %d; target = %d' %
                (dicts['src'].size(), dicts['tgt'].size()))
    logger.info(' * number of training sentences. %d' %
                len(dataset['train']['src']))
    logger.info(' * maximum batch size. %d' % opt.batch_size)

    logger.info('Building model...')

    sent_encoder = neusum.Models.Encoder(opt, dicts['src'])
    doc_encoder = neusum.Models.DocumentEncoder(opt)
    pointer = neusum.Models.Pointer(opt, dicts['tgt'])
    if opt.dec_init == "simple":
        decIniter = neusum.Models.DecInit(opt)
    elif opt.dec_init == "att":
        decIniter = neusum.Models.DecInitAtt(opt)
    else:
        raise ValueError('Unknown decoder init method: {0}'.format(opt.dec_init))

    model = neusum.Models.NMTModel(sent_encoder, doc_encoder, pointer, decIniter, rouge_calculator)
    summarizer = neusum.Summarizer(opt, model, dataset)

    if len(opt.gpus) >= 1:
        model.cuda()
    else:
        model.cpu()

    if opt.freeze_word_vecs_enc:
        logger.warning('Not updating encoder word embedding.')

    for pr_name, p in model.named_parameters():
        logger.info(pr_name)
        # p.data.uniform_(-opt.param_init, opt.param_init)
        if p.dim() == 1:
            # p.data.zero_()
            p.data.normal_(0, math.sqrt(6 / (1 + p.size(0))))
        else:
            xavier_normal(p, math.sqrt(3))
            # xavier_uniform(p)

    sent_encoder.load_pretrained_vectors(opt, logger)

    optim = neusum.Optim(
        opt.optim, opt.learning_rate,
        max_grad_norm=opt.max_grad_norm,
        max_weight_value=opt.max_weight_value,
        lr_decay=opt.learning_rate_decay,
        start_decay_at=opt.start_decay_at,
        decay_bad_count=opt.halve_lr_bad_count
    )

    optim.set_parameters(model.parameters())

    validData = None
    if opt.dev_input_src and opt.dev_ref:
        validData = load_dev_data(summarizer, opt.dev_input_src, opt.dev_ref)
    trainModel(model, summarizer, trainData, validData, dataset, optim)


if __name__ == "__main__":
    main()
