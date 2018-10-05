#!/bin/bash

set -x

for i; do
    echo $i
done

nvidia-smi

sleep 5

export CUDA_VISIBLE_DEVICES=0

DATAHOME=${@:(-2):1}
EXEHOME=${@:(-1):1}

ls -l ${DATAHOME}

ls -l ${EXEHOME}

SAVEPATH=${DATAHOME}/models/neusum

mkdir -p ${SAVEPATH}

cd ${EXEHOME}

python train.py -save_path ${SAVEPATH} \
                -online_process_data \
                -max_doc_len 80 \
                -train_oracle ${DATAHOME}/train/train.rouge_bigram_F1.oracle.F1mmrTrue.regGain \
                -train_src ${DATAHOME}/train/train.txt.src \
                -train_src_rouge ${DATAHOME}/train/train.rouge_bigram_F1.oracle.F1mmrTrue.regGain \
                -src_vocab ${DATAHOME}/train/vocab.txt.100k \
                -train_tgt ${DATAHOME}/train/train.txt.tgt \
                -tgt_vocab ${DATAHOME}/train/vocab.txt.100k \
                -layers 1 -word_vec_size 50 -sent_enc_size 256 -doc_enc_size 256 -dec_rnn_size 256 \
                -sent_brnn -doc_brnn \
                -dec_init simple \
                -att_vec_size 256 \
                -norm_lambda 20 \
                -sent_dropout 0.3 -doc_dropout 0.2 -dec_dropout 0\
                -batch_size 64 -beam_size 1 \
                -epochs 100 \
                -optim adam -learning_rate 0.001 -halve_lr_bad_count 100000 \
                -gpus 0 \
                -curriculum 0 -extra_shuffle \
                -start_eval_batch 1000 -eval_per_batch 1000 \
                -log_interval 100 -log_home ${SAVEPATH} \
                -seed 12345 -cuda_seed 12345 \
                -pre_word_vecs_enc ${DATAHOME}/glove/glove.6B.50d.txt \
                -freeze_word_vecs_enc \
                -dev_input_src ${DATAHOME}/dev/val.txt.src.shuffle.4k \
                -dev_ref ${DATAHOME}/dev/val.txt.tgt.shuffle.4k \
                -max_decode_step 3 -force_max_len
