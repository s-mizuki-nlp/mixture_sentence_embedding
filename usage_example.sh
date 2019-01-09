#!/bin/sh

python ./train_with_rnn.py \
--config_module=config.debug \
--save_dir=./saved_model/ \
--device=cpu \
--verbose

python ./train_with_rnn.py \
--config_module=config.idx_000 \
--save_dir=./saved_model/ \
--device=cuda:0 \
--verbose

python ./train_with_rnn.py \
--config_module=config.ptb_002 \
--save_dir=./saved_model/ \
--device=cuda:3 \
--verbose \
--log_validation_only

# TSUBAME
qsub -g tga-nlp-titech ./train_with_rnn_on_tsubame.sh -c config.debug -d cuda:0 -s ./saved_model/ -l -v