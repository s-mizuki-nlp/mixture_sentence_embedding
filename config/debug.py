#!/usr/bin/env python
# -*- coding:utf-8 -*-

import sys,io,os
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
        "alpha": {
            "n_dim_in":n_dim_lstm_output,
            "n_dim_out":1,
            "n_dim_hidden":n_dim_lstm_output,
            "n_hidden":2,
            "activation_function":torch.relu
        },
        "mu": {
            "n_dim_in":n_dim_lstm_output,
            "n_dim_out":n_dim_latent,
            "n_dim_hidden":n_dim_lstm_output,
            "n_hidden":2,
            "activation_function":torch.relu
        },
        "sigma": {
            "n_dim_in":n_dim_lstm_output,
            "n_dim_out":n_dim_latent,
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
        "param_tau":1.0,
        "expect_log_alpha":False
    },
    "loss": {
        "empirical_wasserstein": {
            "n_slice":10,
            "scale":10.
        },
        "kldiv": {
            "enabled":True,
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
    "corpus":os.path.join(dataset_dir, "wikipedia_en/sample.txt"),
    "size":100000,
    "min_seq_len":20,
    "max_seq_len":80,
    "dictionary":os.path.join(dataset_dir, "wikipedia_en/vocab_wordpiece.dic"),
    "log_file_path":f"log_train_progress_{__name__}.log"
}

## optimizer
cfg_optimizer = {
    "gradient_clip":1.0,
    "n_epoch":10,
    "n_minibatch":10,
    "optimizer":torch.optim.Adam,
    "lr":0.001,
    # "validation_split":0.0,
    "validation_interval":20
}
