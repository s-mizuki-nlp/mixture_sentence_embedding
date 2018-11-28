#!/bin/sh

./train_with_rnn.py \
    --config_module=config.debug \
    --save_dir=./saved_model/ \
    --device=cpu \
    --verbose
