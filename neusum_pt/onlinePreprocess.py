import logging
from ast import literal_eval as make_tuple
import torch
import numpy
import neusum

try:
    import ipdb
except ImportError:
    pass

lower = True
seq_length = 100
max_doc_len = 80
report_every = 100000
shuffle = 1
norm_lambda = 5

logger = logging.getLogger(__name__)


def makeVocabulary(filenames, size):
    vocab = neusum.Dict([neusum.Constants.PAD_WORD, neusum.Constants.UNK_WORD,
                         neusum.Constants.BOS_WORD, neusum.Constants.EOS_WORD], lower=lower)
    for filename in filenames:
        with open(filename, encoding='utf-8') as f:
            for sent in f.readlines():
                for word in sent.strip().split(' '):
                    vocab.add(word)

    originalSize = vocab.size()
    vocab = vocab.prune(size)
    logger.info('Created dictionary of size %d (pruned from %d)' %
                (vocab.size(), originalSize))

    return vocab


def initVocabulary(name, dataFiles, vocabFile, vocabSize):
    vocab = None
    if vocabFile is not None:
        # If given, load existing word dictionary.
        logger.info('Reading ' + name + ' vocabulary from \'' + vocabFile + '\'...')
        vocab = neusum.Dict(lower=lower)
        vocab.loadFile(vocabFile)
        logger.info('Loaded ' + str(vocab.size()) + ' ' + name + ' words')

    if vocab is None:
        # If a dictionary is still missing, generate it.
        logger.info('Building ' + name + ' vocabulary...')
        genWordVocab = makeVocabulary(dataFiles, vocabSize)

        vocab = genWordVocab

    return vocab


def saveVocabulary(name, vocab, file):
    logger.info('Saving ' + name + ' vocabulary to \'' + file + '\'...')
    vocab.writeFile(file)


def np_softmax(x, a=1):
    """Compute softmax values for each sets of scores in x."""
    return numpy.exp(a * x) / numpy.sum(numpy.exp(a * x), axis=0)


def makeData(srcFile, tgtFile, train_oracle_file, train_src_rouge_file, srcDicts, tgtDicts):
    src, tgt = [], []
    src_raw = []
    src_rouge = []
    oracle = []
    sizes = []
    count, ignored = 0, 0

    logger.info('Processing %s & %s ...' % (srcFile, tgtFile))
    srcF = open(srcFile, encoding='utf-8')
    tgtF = open(tgtFile, encoding='utf-8')
    oracleF = open(train_oracle_file, encoding='utf-8')
    src_rougeF = open(train_src_rouge_file, encoding='utf-8')

    while True:
        sline = srcF.readline()
        tline = tgtF.readline()
        oline = oracleF.readline()
        src_rouge_line = src_rougeF.readline()

        # normal end of file
        if sline == "" and tline == "":
            break

        # source or target does not have same number of lines
        if sline == "" or tline == "" or src_rouge_line == "":
            logger.info('WARNING: source and target do not have the same number of sentences')
            break

        sline = sline.strip()
        tline = tline.strip()
        oline = oline.strip()
        src_rouge_line = src_rouge_line.strip()

        # source and/or target are empty
        if sline == "" or tline == "" or ('None' in oline) or ('nan' in src_rouge_line):
            logger.info('WARNING: ignoring an empty line (' + str(count + 1) + ')')
            continue

        srcSents = sline.split('##SENT##')[:max_doc_len]
        tgtSents = tline.split('##SENT##')
        rouge_gains = src_rouge_line.split('\t')[1:]
        srcWords = [x.split(' ')[:seq_length] for x in srcSents]
        tgtWords = ' '.join(tgtSents)
        oracle_combination = make_tuple(oline.split('\t')[0])
        # oracle_combination = [(x + 1) for x in oracle_combination] + [0]
        oracle_combination = [x for x in oracle_combination]  # no sentinel

        index_out_of_range = [x >= max_doc_len for x in oracle_combination]
        if any(index_out_of_range):
            logger.info('WARNING: oracle exceeds max_doc_len, ignoring (' + str(count + 1) + ')')
            continue

        src_raw.append(srcSents)

        src.append([srcDicts.convertToIdx(word,
                                          neusum.Constants.UNK_WORD) for word in srcWords])
        tgt.append(tgtWords)

        oracle.append(torch.LongTensor(oracle_combination))
        rouge_gains = [[float(gain) for gain in x.split(' ')] for x in rouge_gains]
        # rouge_gains = [torch.FloatTensor(x) for x in rouge_gains]
        # rouge_gains = [(x - torch.min(x)) / (torch.max(x) - torch.min(x)) for x in rouge_gains][:1]
        rouge_gains = [numpy.array(x) for x in rouge_gains]
        rouge_gains = [(x - numpy.min(x)) / (numpy.max(x) - numpy.min(x)) for x in rouge_gains]
        rouge_gains = [torch.from_numpy(np_softmax(x, norm_lambda)).float() for x in rouge_gains]
        src_rouge.append(rouge_gains)

        sizes += [len(srcWords)]

        count += 1

        if count % report_every == 0:
            logger.info('... %d sentences prepared' % count)

    srcF.close()
    tgtF.close()
    oracleF.close()
    src_rougeF.close()

    if shuffle == 1:
        logger.info('... shuffling sentences')
        perm = torch.randperm(len(src))
        src = [src[idx] for idx in perm]
        src_raw = [src_raw[idx] for idx in perm]
        tgt = [tgt[idx] for idx in perm]
        oracle = [oracle[idx] for idx in perm]
        src_rouge = [src_rouge[idx] for idx in perm]
        sizes = [sizes[idx] for idx in perm]

    logger.info('... sorting sentences by size')
    _, perm = torch.sort(torch.Tensor(sizes))
    src = [src[idx] for idx in perm]
    src_raw = [src_raw[idx] for idx in perm]
    tgt = [tgt[idx] for idx in perm]
    oracle = [oracle[idx] for idx in perm]
    src_rouge = [src_rouge[idx] for idx in perm]

    logger.info('Prepared %d sentences (%d ignored due to length == 0 or > %d)' %
                (len(src), ignored, seq_length))
    return src, src_raw, tgt, oracle, src_rouge


def prepare_data_online(train_src, src_vocab, train_tgt, tgt_vocab, train_oracle, train_src_rouge):
    dicts = {}
    dicts['src'] = initVocabulary('source', [train_src], src_vocab, 0)
    dicts['tgt'] = initVocabulary('target', [train_tgt], tgt_vocab, 0)

    logger.info('Preparing training ...')
    train = {}
    train['src'], train['src_raw'], train['tgt'], \
    train['oracle'], train['src_rouge'] = makeData(train_src, train_tgt,
                                                   train_oracle, train_src_rouge,
                                                   dicts['src'], dicts['tgt'])

    dataset = {'dicts': dicts,
               'train': train,
               # 'valid': valid
               }
    return dataset
