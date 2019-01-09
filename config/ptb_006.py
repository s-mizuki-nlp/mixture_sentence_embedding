#!/usr/bin/env python
# -*- coding:utf-8 -*-

import sys,io,os
import warnings
import torch
import torch.optim

highway = False
bidirectional = True
n_dim_latent = 128
n_dim_lstm_hidden = 128
n_dim_embedding = 128
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
        # you can disable predicting \alpha by just specifying None
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
            # n_dim_out can be either 1 or n_dim_latent.
            # "n_dim_out":n_dim_latent,
            "n_dim_out":1,
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
        "n_sample":16, # n_gmm_component,
        "param_tau":0.1,
        "expect_log_alpha":True,
        "enable_gumbel_softmax_trick":True
    },
    "loss": {
        "reg": {
            "empirical_sliced_wasserstein": {
                "n_slice":100,
                "scale":1.0
            }
        },
        "kldiv": {
            "enabled":False,
            "scale":0.0
        }
    },
    "prior": {
        "n_gmm_component":n_gmm_component,
        "n_dim":n_dim_latent,
        "expected_wd":2.0
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
        "corpus":os.path.join(dataset_dir, "language_modeling/ptb_mikolov/train.txt"),
        "size":42068,
        "min_seq_len":None,
        "max_seq_len":None
    },
    "test":{
        "corpus":os.path.join(dataset_dir, "language_modeling/ptb_mikolov/test.txt"),
        "min_seq_len":None,
        "max_seq_len":None,
        "evaluation_metrics":["kldiv_ana","kldiv_mc"]
    },
    "dictionary":os.path.join(dataset_dir, "language_modeling/ptb_mikolov/vocab.dic"),
    "log_file_path":f"log_train_progress_{__name__}.log"
}

## optimizer
cfg_optimizer = {
    "gradient_clip":1.0,
    "n_epoch":100,
    "n_minibatch":128,
    "optimizer":torch.optim.Adam,
    "lr":0.001,
    "validation_interval":100
}

## assertion
if cfg_auto_encoder["encoder"]["alpha"] is None:
    if cfg_auto_encoder["loss"]["kldiv"]["enabled"]:
        raise ValueError("you should disable kullback-leibler divergence when you don't predict alpha.")
if not cfg_auto_encoder["sampler"]["enable_gumbel_softmax_trick"]:
    warnings.warn("disabling gumbel-softmax trick will result in improper sampling from posterior distribution. ARE YOU OK?")

# regularizer
if "empirical_sliced_wasserstein" in cfg_auto_encoder["loss"]["reg"]:
    if "sinkhorn_wasserstein" in cfg_auto_encoder["loss"]["reg"]:
        raise ValueError("you can't define different regularizers simultaneously.")
    else:
        pass
else:
    if not "sinkhorn_wasserstein" in cfg_auto_encoder["loss"]["reg"]:
        raise ValueError("you have to define wasserstein regularizer.")

if "sinkhorn_wasserstein" in cfg_auto_encoder["loss"]["reg"]:
    if cfg_auto_encoder["encoder"]["sigma"]["n_dim_out"] == 1:
        warnings.warn("diagonal covariance is recommended. ARE YOU OK?")
