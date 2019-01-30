#!/usr/bin/env python
# -*- coding:utf-8 -*-

import os, sys, io
from copy import deepcopy
import argparse
from contextlib import ExitStack
import importlib
from typing import List, Dict, Union, Any, Optional

import numpy as np
import progressbar
import pprint

import torch
from torch import nn

wd = os.path.dirname(__file__)
wd = "." if wd == "" else wd
os.chdir(wd)

from common.loader.text import TextLoader
from preprocess.tokenizer import CharacterTokenizer
from preprocess.corpora import Dictionary
from preprocess.dataset_feeder import GeneralSentenceFeeder
from preprocess import utils

# encoders
from model.multi_layer import MultiDenseLayer, IdentityLayer
from model.encoder import GMMLSTMEncoder
# decoder
from model.attention import  SimpleGlobalAttention, MultiHeadedAttention, PassTuru
from model.decoder import SelfAttentiveLSTMDecoder
from model.decoder import SimplePredictor
# regularizers
## sampler
from model.noise_layer import GMMSampler
## prior distribution
from distribution.mixture import MultiVariateGaussianMixture
from utility import generate_random_orthogonal_vectors, calculate_prior_dist_params, calculate_mean_l2_between_sample
## loss functions
from model.loss import EmpiricalSlicedWassersteinDistance, GMMSinkhornWassersteinDistance, GMMApproxKLDivergence
from model.loss import MaskedKLDivLoss
from model.loss import PaddedNLLLoss
# variational autoencoder
from model.vae import VariationalAutoEncoder
## used for evaluation
from utility import enumerate_optional_metrics, write_log_and_progress

# estimator
from train import Estimator


def _parse_args():

    parser = argparse.ArgumentParser(description="Wasserstein AutoEncoder using Gaussian Mixture and RNN Encoder/Decoder: train/validation script")
    parser.add_argument("--config_module", "-c", required=True, type=str, help="config module name. example: `config.default`")
    parser.add_argument("--save_dir", "-s", required=True, type=str, help="directory for saving trained model")
    parser.add_argument("--device", "-d", required=False, type=str, default="cpu", help="computing device. DEFAULT:cpu")
    parser.add_argument("--gpus", required=False, type=str, default="0", help="GPU device ids to be used for dataparallel processing. DEFAULT:0")
    parser.add_argument("--save_every_epoch", action="store_true", help="save trained model at every epoch. DEFAULT:False")
    parser.add_argument("--log_validation_only", action="store_true", help="record validation metrics only. DEFAULT:False")
    parser.add_argument("--verbose", action="store_true", help="output verbosity")
    args = parser.parse_args()

    if args.device.find("cuda") != -1:
        assert torch.cuda.is_available(), "GPU is unavailable but cuda device was specified."
        args.gpus = [int(i) for i in args.gpus.split(",")]
    else:
        args.gpus = []
    args.device = torch.device(args.device)

    return args


def main():

    args = _parse_args()

    # import configuration
    config = importlib.import_module(name=args.config_module)
    file_name_suffix = args.config_module

    cfg_auto_encoder = config.cfg_auto_encoder
    cfg_corpus = config.cfg_corpus
    cfg_optimizer = config.cfg_optimizer

    # show important configurations
    print("corpus:")
    pprint.pprint(cfg_corpus)
    print("optimization:")
    for k, v in cfg_optimizer.items():
        print(f"\t{k}:{v}")
    print("prior distribution:")
    pprint.pprint(cfg_auto_encoder["prior"])

    # instanciate corpora
    tokenizer = CharacterTokenizer()
    dictionary = Dictionary.load(file_path=cfg_corpus["dictionary"])
    bos, eos = list(dictionary.special_tokens)

    dict_data_feeder = {}
    for corpus_type in "train,dev,test".split(","):
        if not corpus_type in cfg_corpus:
            continue
        cfg_corpus_t = cfg_corpus[corpus_type]
        corpus_t = TextLoader(file_path=cfg_corpus_t["corpus"])
        data_feeder_t = GeneralSentenceFeeder(corpus = corpus_t,
                                              tokenizer = tokenizer, dictionary = dictionary,
                                              n_minibatch=cfg_optimizer["n_minibatch"], validation_split=0.,
                                              min_seq_len=cfg_corpus_t["min_seq_len"],
                                              max_seq_len=cfg_corpus_t["max_seq_len"],
                                              # append `<eos>` at the end of each sentence
                                              bos_symbol=None, eos_symbol=eos)
        dict_data_feeder[corpus_type] = data_feeder_t

    # setup logger
    dict_logger = {}
    for phase in "train,test".split(","):
        path_log_file = cfg_corpus["log_file_path"] + f".{phase}"
        if os.path.exists(path_log_file):
            os.remove(path_log_file)
        logger = io.open(path_log_file, mode="w")
        dict_logger[phase] = logger

    # instanciate variational autoencoder

    ## prior distribution
    n_dim_gmm = cfg_auto_encoder["prior"]["n_dim"]
    n_prior_gmm_component = cfg_auto_encoder["prior"]["n_gmm_component"]

    ## regularizer type
    regularizer_name = next(iter(cfg_auto_encoder["loss"]["reg"]))
    print(f"regularizer type: {regularizer_name}")
    is_sliced_wasserstein = regularizer_name.find("sliced_wasserstein") != -1

    # calculate l2 norm and stdev of the mean and stdev of prior distribution
    expected_wd = cfg_auto_encoder["prior"].get("expected_wd", 1.0)
    l2_norm, std = calculate_prior_dist_params(expected_wd=expected_wd, n_dim_latent=n_dim_gmm, sliced_wasserstein=is_sliced_wasserstein)
    ## overwrite auto values with user-defined values
    l2_norm = cfg_auto_encoder["prior"].get("l2_norm", l2_norm)
    std = cfg_auto_encoder["prior"].get("std", std)
    print("prior distribution parameters.")
    print(f"\tl2_norm:{l2_norm:2.3f}, stdev:{std:2.3f}")
    vec_alpha = np.full(shape=n_prior_gmm_component, fill_value=1./n_prior_gmm_component)
    mat_mu = generate_random_orthogonal_vectors(n_dim=n_dim_gmm, n_vector=n_prior_gmm_component, l2_norm=l2_norm)
    vec_std = np.ones(shape=n_prior_gmm_component) * std
    prior_distribution = MultiVariateGaussianMixture(vec_alpha=vec_alpha, mat_mu=mat_mu, vec_std=vec_std)
    path_prior = os.path.join(args.save_dir, f"prior_distribution.gmm.{file_name_suffix}.pickle")
    prior_distribution.save(file_path=path_prior)

    # instanciate variational autoencoder
    cfg_encoder = cfg_auto_encoder["encoder"]
    ## MLP for \alpha, \mu, \sigma
    for param_name in "alpha,mu,sigma".split(","):
        if cfg_encoder[param_name] is None:
            if param_name == "alpha":
                layer = None
            elif param_name == "sigma":
                layer = IdentityLayer(n_dim_out=cfg_auto_encoder["prior"]["n_dim"])
            else:
                pprint.pprint(cfg_encoder)
                raise NotImplementedError("unsupported configuration detected.")
        else:
            layer = MultiDenseLayer(**cfg_encoder[param_name])
        cfg_encoder["lstm"][f"encoder_{param_name}"] = layer
    ## encoder
    encoder = GMMLSTMEncoder(n_vocab=dictionary.max_id+1, device=args.device, **cfg_encoder["lstm"])

    ## sampler(from posterior)
    sampler = GMMSampler(device=args.device, **cfg_auto_encoder["sampler"])

    ## decoder
    latent_decoder_name = next(iter(cfg_auto_encoder["decoder"]["latent"]))
    if latent_decoder_name == "simple_attention":
        latent_decoder = SimpleGlobalAttention(**cfg_auto_encoder["decoder"]["latent"][latent_decoder_name])
    elif latent_decoder_name == "multi_head_attention":
        latent_decoder = MultiHeadedAttention(**cfg_auto_encoder["decoder"]["latent"][latent_decoder_name])
    elif latent_decoder_name == "pass_turu":
        latent_decoder = PassTuru(**cfg_auto_encoder["decoder"]["latent"][latent_decoder_name])
    else:
        raise NotImplementedError(f"unsupported latent decoder:{latent_decoder_name}")
    decoder = SelfAttentiveLSTMDecoder(latent_decoder=latent_decoder, device=args.device, **cfg_auto_encoder["decoder"]["lstm"])
    ## prediction layer
    predictor = SimplePredictor(n_dim_out=dictionary.max_id+1, log=True, **cfg_auto_encoder["predictor"])

    ## variational autoencoder
    model = VariationalAutoEncoder(seq_to_gmm_encoder=encoder, gmm_sampler=sampler,
                                   set_to_state_decoder=decoder, state_to_seq_decoder=predictor)
    if len(args.gpus) > 1:
        model = nn.DataParallel(model, device_ids = args.gpus)
    model.to(device=args.device)

    ## loss layers
    ### regularizer between posteriors and prior; wasserstein distance or kullback-leibler divergence: d(p(z|x), q(z))
    cfg_regularizer = cfg_auto_encoder["loss"]["reg"]
    if regularizer_name == "empirical_sliced_wasserstein":
        loss_regularizer = EmpiricalSlicedWassersteinDistance(device=args.device, **cfg_regularizer["empirical_sliced_wasserstein"])
    elif regularizer_name == "sinkhorn_wasserstein":
        loss_regularizer = GMMSinkhornWassersteinDistance(device=args.device, **cfg_regularizer["sinkhorn_wasserstein"])
    elif regularizer_name == "kullback_leibler":
        loss_regularizer = GMMApproxKLDivergence(device=args.device, **cfg_regularizer["kullback_leibler"])
    else:
        raise NotImplementedError("unsupported regularizer type:", regularizer_name)

    ### kullback-leibler divergence on \alpha: KL(p(\alpha|x), q(\alpha))
    if cfg_auto_encoder["loss"]["kldiv"]["enabled"]:
        loss_kldiv = MaskedKLDivLoss(scale=cfg_auto_encoder["loss"]["kldiv"]["scale"], reduction="samplewise_mean")
    else:
        loss_kldiv = None
    ### negtive log likelihood: -lnp(x|z); z~p(z|x)
    loss_reconst = PaddedNLLLoss(reduction="samplewise_mean")

    ### instanciate estimator ###
    estimator = Estimator(model=model, loss_reconst=loss_reconst, loss_layer_reg=loss_regularizer, loss_layer_kldiv=loss_kldiv,
                          device=args.device, verbose=args.verbose)

    # optimizer for variational autoencoder
    optimizer = cfg_optimizer["optimizer"](model.parameters(), lr=cfg_optimizer["lr"])

    # start training
    n_epoch_total = cfg_optimizer["n_epoch"]
    # iterate over epoch
    n_iteration = 0
    n_processed = 0
    for n_epoch in range(n_epoch_total):
        print(f"epoch:{n_epoch}")

        #### train phase ####
        phase = "train"
        print(f"phase:{phase}")
        model.train()
        logger = dict_logger[phase]
        cfg_corpus_t = cfg_corpus[phase]
        lst_eval_metrics = enumerate_optional_metrics(cfg_metrics=cfg_corpus[phase].get("evaluation_metrics",[]), n_epoch=n_epoch+1)
        q = progressbar.ProgressBar(max_value=cfg_corpus_t["size"])
        n_progress = 0
        q.update(n_progress)
        ## iterate over mini-batch
        for train, _ in dict_data_feeder[phase]:
            n_iteration += 1

            # update scale parameter of wasserstein distance layer
            estimator.reg_wasserstein.update_scale_parameter(n_processed=n_processed)
            # update annealing parameter of sampler layer
            sampler.update_anneal_parameter(n_processed=n_processed)

            # training
            if cfg_optimizer["validation_interval"] is not None:
                train_mode = not(n_iteration % cfg_optimizer["validation_interval"] == 0)
            else:
                train_mode = True
            lst_seq_len, lst_seq = utils.len_pad_sort(lst_seq=train)

            metrics_batch = estimator.train_single_step(lst_seq=lst_seq, lst_seq_len=lst_seq_len, optimizer=optimizer,
                                                        prior_distribution=prior_distribution,
                                                        clip_gradient_value=cfg_optimizer["gradient_clip"],
                                                        evaluation_metrics=lst_eval_metrics)

            n_processed += len(lst_seq_len)
            n_progress += len(lst_seq_len)

            # logging and reporting
            write_log_and_progress(n_epoch=n_epoch,n_processed=n_processed,
                                   mode="train" if train_mode else "val",
                                   dict_metrics=metrics_batch,
                                   logger = logger,
                                   output_log= not(args.log_validation_only) or not(train_mode),
                                   output_std=args.verbose
                                   )

            # next iteration
            q.update(n_progress)

        # save progress
        if args.save_every_epoch:
            path_trained_model_e = os.path.join(args.save_dir, f"lstm_vae.{file_name_suffix}.model." + str(n_epoch))
            print(f"saving...:{path_trained_model_e}")
            torch.save(model.state_dict(), path_trained_model_e)

        #### (optional) update prior distribution ###
        if "update" in cfg_auto_encoder["prior"]:
            cfg_update_prior = cfg_auto_encoder["prior"]["update"]
            if n_epoch in cfg_update_prior["target_epoch"]:
                print("update prior distribution. wait for a while...")
                prior_distribution_new = estimator.train_prior_distribution(
                    cfg_optimizer=cfg_update_prior["optimizer"],
                    cfg_regularizer=cfg_update_prior["regularizer"],
                    prior_distribution=prior_distribution,
                    data_feeder=dict_data_feeder["train"]
                )
                # renew prior distribution
                prior_distribution = prior_distribution_new
                prior_distribution.save(file_path=path_prior)
        else:
            print("we do not update prior distribution. skip training.")

        #### test phase ####
        if not "test" in dict_data_feeder:
            print("we do not have testset. skip evaluation.")
            continue

        phase = "test"
        print(f"phase:{phase}")
        model.eval()
        logger = dict_logger[phase]
        lst_eval_metrics = enumerate_optional_metrics(cfg_metrics=cfg_corpus[phase].get("evaluation_metrics",[]), n_epoch=n_epoch+1)
        lst_metrics_batch = []
        ## iterate over mini-batch
        for batch, _ in dict_data_feeder[phase]:
            lst_seq_len, lst_seq = utils.len_pad_sort(lst_seq=batch)

            metrics_batch = estimator.test_single_step(lst_seq=lst_seq, lst_seq_len=lst_seq_len,
                                                       prior_distribution=prior_distribution,
                                                       evaluation_metrics=lst_eval_metrics)
            lst_metrics_batch.append(metrics_batch)

        # calculate whole metrics
        metrics = {}
        vec_n_sentence = np.array([m["n_sentence"] for m in lst_metrics_batch])
        vec_n_token = np.array([m["n_token"] for m in lst_metrics_batch])
        metrics["n_sentence"] = np.sum(vec_n_sentence)
        metrics["n_token"] = np.sum(vec_n_token)
        metrics["n_token_per_sentence"] = metrics["n_token"] / metrics["n_sentence"]
        for metric_name in lst_metrics_batch[0].keys():
            vec_values = np.array([np.nan if m[metric_name] is None else m[metric_name] for m in lst_metrics_batch])
            if metric_name in ["n_sentence","n_token"]:
                continue
            elif metric_name.startswith("mean_"): # sentence-wise mean
                metrics[metric_name] = np.sum(vec_n_sentence * vec_values) / np.sum(vec_n_sentence)
            elif metric_name == "nll_token": # token-wise mean
                metrics[metric_name] = np.sum(vec_n_token * vec_values) / np.sum(vec_n_token)
            else: # token-wise mean * average sentence length
                metrics[metric_name] = np.sum(vec_n_sentence * vec_values) * metrics["n_token_per_sentence"] / metrics["n_token"]

        # logging and reporting
        write_log_and_progress(n_epoch=n_epoch, n_processed=n_processed, mode="test", dict_metrics=metrics,
                               logger=logger, output_log=True, output_std=True)

        ### proceed to next epoch ###

    # end of epoch
    for logger in dict_logger.values():
        logger.close()

    # save trained model
    path_trained_model_e = os.path.join(args.save_dir, f"lstm_vae.{file_name_suffix}.model." + str(n_epoch_total))
    print(f"saving...:{path_trained_model_e}")
    torch.save(model.state_dict(), path_trained_model_e)


if __name__ == "__main__":
    main()
    print("finished. good-bye.")
