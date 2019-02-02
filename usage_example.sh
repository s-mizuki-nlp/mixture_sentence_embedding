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

python ./train_with_rnn.py \
--config_module=config.wikitext_003 \
--save_dir=./saved_model/ \
--device=cuda:2 \
--verbose \
--log_validation_only \
--save_every_epoch

# TSUBAME
qsub -g tga-nlp-titech ./train_with_rnn_on_tsubame.sh -c config.debug -d cuda:0 -s ./saved_model/ -l -v

# restart from checkpoint
python ./train_with_rnn.py \
--config_module=config.wikitext_002 \
--save_dir=./temp/ \
--device=cpu \
--verbose \
--log_validation_only \
--save_every_epoch \
--checkpoint=./saved_model/lstm_vae.config.wikitext_002.model.0 \
--checkpoint_epoch=1 \
--checkpoint_prior=./saved_model/prior_distribution.gmm.config.wikitext_002.pickle