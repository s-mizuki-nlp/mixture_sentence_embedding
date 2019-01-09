#!/usr/bin/env python
# -*- coding:utf-8 -*-

import sys,io,os
import warnings
import torch
import torch.optim

highway = False
bidirectional = True
n_dim_latent = 32
n_head = 4
n_dim_lstm_hidden = 128
n_dim_embedding = 128
n_dim_lstm_output = n_dim_lstm_hidden * (bidirectional + 1)
n_gmm_component = 8

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
        #"alpha": {
        #   "n_dim_in":n_dim_lstm_output,
        #   "n_dim_out":1,
        #   "n_dim_hidden":n_dim_lstm_output,
        #   "n_hidden":2,
        #   "activation_function":torch.relu
        #},
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
            "n_dim_out":n_dim_latent, # it must be either 1 or n_dim_latent.
            "n_dim_hidden":n_dim_lstm_output,
            "n_hidden":2,
            "activation_function":torch.relu
        }
    },
    "decoder": {
        "lstm":{
            "n_dim_lstm_hidden":n_dim_lstm_hidden,
            "n_dim_lstm_input":n_dim_lstm_hidden
        },
        "latent":{
            # you can choose one of these latent representation decoders:
            # 1) simple attention
            # 2) multi-head attention (not implemented yet)
            "multi_head_attention":{
                "n_head":n_head,
                "n_dim_query":n_dim_lstm_hidden*2,
                "n_dim_memory":n_dim_latent,
                "n_dim_out":n_dim_lstm_hidden,
                "dropout":0.0,
                "transform_memory_bank":False
            }
        }
    },
    "predictor": {
        "n_dim_in":n_dim_lstm_hidden
    },
    "sampler": {
        "n_sample":4,
        "param_tau":0.03,
        "expect_log_alpha":True,
        "enable_gumbel_softmax_trick":True
    },
    "loss": {
        "reg": {
            # you can choose one of these regularizers:
            # 1) empirical sliced wasserstein distance: d(E_x[p(z|x)],q(z))
            # "empirical_sliced_wasserstein": {
            #     "n_slice":10,
            #     "scale":1.
            # },
            # 2) sinkhorn wasserstein distance: d(p(z|x),q(z))
            "sinkhorn_wasserstein": {
                "sinkhorn_lambda":0.1,
                "sinkhorn_iter_max":100,
                "sinkhorn_threshold":0.1,
                "scale":1.0
            }
        },
        "kldiv": {
            "enabled":False,
            "scale":1.
        }
    },
    "prior": {
        "n_gmm_component":n_gmm_component,
        "n_dim":n_dim_latent,
        "expected_wd":2.0,
        # if you want, you can manually specify l2 norm and standard deviation
        "l2_norm": 1.0,
        "std": 0.8
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
        "evaluation_metrics":["kldiv_ana"]
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
# MLP_{\alpha} vs KL(p(\alpha|x) || q(\alpha))
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

if "simple_attention" in cfg_auto_encoder["decoder"]["latent"]:
    cfg_attn = cfg_auto_encoder["decoder"]["latent"]["simple_attention"]
    cfg_lstm = cfg_auto_encoder["decoder"]["lstm"]
    assert cfg_attn["n_dim_query"] == cfg_lstm["n_dim_lstm_hidden"] * 2, "query size must be the double of hidden state."
    assert cfg_attn["n_dim_memory"] == cfg_lstm["n_dim_lstm_input"], "you can't change memory dimension size using simple_attention."

if "multi_head_attention" in cfg_auto_encoder["decoder"]["latent"]:
    cfg_attn = cfg_auto_encoder["decoder"]["latent"]["multi_head_attention"]
    cfg_lstm = cfg_auto_encoder["decoder"]["lstm"]
    if not cfg_attn["transform_memory_bank"]:
        assert cfg_attn["n_dim_memory"] == n_dim_latent, "memory bank must be same size as latent dimension."
    assert cfg_attn["n_dim_query"] == cfg_lstm["n_dim_lstm_hidden"] * 2, "query size must be the double of hidden state."
    assert cfg_attn["n_dim_out"] == cfg_lstm["n_dim_lstm_input"], "output of the attention layer must be same size as input of the lstm layer."
