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
