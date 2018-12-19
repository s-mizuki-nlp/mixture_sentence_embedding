#!/usr/bin/env python
# -*- coding:utf-8 -*-

import sys,io,os
import warnings
import torch
import torch.optim

highway = False
bidirectional = True
n_dim_latent = 16
n_dim_lstm_hidden = 16
n_dim_embedding = 16
n_dim_lstm_output = n_dim_lstm_hidden * (bidirectional + 1)
n_gmm_component = 4

cfg_auto_encoder = {
    "encoder": {
        "lstm": {
            "n_dim_embedding":n_dim_embedding,
            "n_dim_lstm_hidden":n_dim_lstm_hidden,
            "n_lstm_layer":2,
            "custom_embedding_layer":None,
            "bidirectional":bidirectional,
            "highway":highway
        },
        # if you want to disable predicting \alpha, just specify None
        # "alpha": {
        #     "n_dim_in":n_dim_lstm_output,
        #     "n_dim_out":1,
        #     "n_dim_hidden":n_dim_lstm_output,
        #     "n_hidden":2,
        #     "activation_function":torch.relu
        # },
        "alpha": None,
        "mu": {
            "n_dim_in":n_dim_lstm_output,
            "n_dim_out":n_dim_latent,
            "n_dim_hidden":n_dim_lstm_output,
            "n_hidden":2,
            "activation_function":torch.relu
        },
        "sigma": {
            "n_dim_in":n_dim_lstm_output,
            "n_dim_out":1, # it must be either 1 or n_dim_latent.
            "n_dim_hidden":n_dim_lstm_output,
            "n_hidden":2,
            "activation_function":torch.relu
        }
    },
    "decoder": {
        "n_dim_lstm_hidden":n_dim_lstm_hidden,
        "n_dim_memory":n_dim_latent,
        "custom_attention_layer":None
    },
    "predictor": {
        "n_dim_in":n_dim_lstm_hidden
    },
    "sampler": {
        "n_sample":n_gmm_component,
        "param_tau":0.1,
        "expect_log_alpha":True,
        "enable_gumbel_softmax_trick":True
    },
    "loss": {
        "empirical_wasserstein": {
            "n_slice":10,
            "scale":1.
        },
        "kldiv": {
            "enabled":False,
            "scale":1.
        }
    },
    "prior": {
        "n_gmm_component":n_gmm_component,
        "n_dim":n_dim_latent,
        "expected_swd":2.0
    }
}

## corpus location
_hostname = os.uname()[1]
if _hostname == "Ubuntu-Precision-Tower-3420":
    dataset_dir = "/home/sakae/Windows/dataset/"
elif _hostname == "iris":
    dataset_dir = "/home/sakae/dataset/"
else:
    raise NotImplementedError(f"unknown environment:{_hostname}")
print(f"dataset directory:{dataset_dir}")
cfg_corpus = {
    "train":{
        "corpus":os.path.join(dataset_dir, "wikipedia_en/sample_train.txt"),
        "size":1000,
        "min_seq_len":20,
        "max_seq_len":80
    },
    "test":{
        "corpus":os.path.join(dataset_dir, "wikipedia_en/sample_test.txt"),
        "size":100,
        "min_seq_len":None,
        "max_seq_len":None
    },
    "dictionary":os.path.join(dataset_dir, "wikipedia_en/vocab_wordpiece.dic"),
    "log_file_path":f"log_train_progress_{__name__}.log"
}

## optimizer
cfg_optimizer = {
    "gradient_clip":1.0,
    "n_epoch":10,
    "n_minibatch":30,
    "optimizer":torch.optim.Adam,
    "lr":0.001,
    "validation_interval":20
}

## assertion
if cfg_auto_encoder["encoder"]["alpha"] is None:
    if cfg_auto_encoder["loss"]["kldiv"]["enabled"]:
        raise ValueError("you should disable kullback-leibler divergence when you don't predict alpha.")
if not cfg_auto_encoder["sampler"]["enable_gumbel_softmax_trick"]:
    warnings.warn("disabling gumbel-softmax trick will result in improper sampling from posterior distribution. ARE YOU OK?")