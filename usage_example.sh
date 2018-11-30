#!/bin/sh

./train_with_rnn.py \
--config_module=config.idx_000 \
--save_dir=./saved_model/ \
--device=cpu \
--verbose
