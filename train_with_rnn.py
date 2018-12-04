#!/usr/bin/env python
# -*- coding:utf-8 -*-

import os, sys, io
import argparse
from contextlib import ExitStack
import importlib
from typing import List, Dict, Union, Any

import numpy as np
import pickle
import progressbar

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
from model.multi_layer import MultiDenseLayer
from model.encoder import GMMLSTMEncoder
# decoder
from model.decoder import SelfAttentiveLSTMDecoder
from model.decoder import SimplePredictor
# regularizers
## sampler
from model.noise_layer import GMMSampler
## prior distribution
from distribution.mixture import MultiVariateGaussianMixture
from utility import generate_random_orthogonal_vectors, calculate_prior_dist_params
## loss functions
from model.loss import EmpiricalSlicedWassersteinDistance
from model.loss import MaskedKLDivLoss
from model.loss import PaddedNLLLoss
# variational autoencoder
from model.vae import VariationalAutoEncoder


def _parse_args():

    parser = argparse.ArgumentParser(description="Wasserstein AutoEncoder using Gaussian Mixture and RNN Encoder/Decoder: train/validation script")
    parser.add_argument("--config_module", "-c", required=True, type=str, help="config module name. example: `config.default`")
    parser.add_argument("--save_dir", "-s", required=True, type=str, help="directory for saving trained model")
    parser.add_argument("--device", "-d", required=False, type=str, default="cpu", help="computing device. DEFAULT:cpu")
    parser.add_argument("--gpus", required=False, type=str, default="0", help="GPU device ids to be used for dataparallel processing. DEFAULT:0")
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


def main_minibatch(model, optimizer, prior_distribution, loss_reconst, loss_reg_wd, loss_reg_kldiv, lst_seq, lst_seq_len,
                   device, train_mode, cfg_auto_encoder, cfg_optimizer):

    if train_mode:
        optimizer.zero_grad()

    with ExitStack() as context_stack:
        # if not train mode, enter into no_grad() context
        if not train_mode:
            context_stack.enter_context(torch.no_grad())

        # create encoder input and decoder output(=ground truth)
        ## omit last `<eos>` symbol from the input sequence
        x_in_len = [seq_len - 1 for seq_len in lst_seq_len]
        x_in = [seq[:-1] for seq in lst_seq]
        x_out_len = lst_seq_len
        x_out = lst_seq

        # convert to torch.tensor
        ## input
        v_x_in = torch.tensor(x_in, dtype=torch.long).to(device=device)
        v_x_in_len = torch.tensor(x_in_len, dtype=torch.long).to(device=device)
        v_x_in_mask = (v_x_in > 0).float().to(device=device)
        ## output
        v_x_out = torch.tensor(x_out, dtype=torch.long).to(device=device)
        v_x_out_len = torch.tensor(x_out_len, dtype=torch.long).to(device=device)

        ## empirical prior distribution
        n_sample = len(x_in) * cfg_auto_encoder["sampler"]["n_sample"]
        v_z_prior = prior_distribution.random(size=n_sample)
        v_z_prior = torch.tensor(v_z_prior, dtype=torch.float32, requires_grad=False).to(device=device)

        ## uniform distribution over $\alpha$
        lst_arr_unif = [np.full(n, 1./n, dtype=np.float32) for n in x_in_len]
        arr_unif = utils.pad_numpy_sequence(lst_arr_unif)
        v_alpha_unif = torch.from_numpy(arr_unif).to(device=device)

        # forward computation of the VAE model
        v_alpha, v_mu, v_sigma, v_z_posterior, v_ln_prob_y = model.forward(x_seq=v_x_in, x_seq_len=v_x_in_len,
                                                                           decoder_max_step=max(x_out_len))
        # regularization losses(sample-wise mean)
        ## empirical sliced wasserstein distance
        v_z_posterior = v_z_posterior.view((-1, cfg_auto_encoder["prior"]["n_dim"]))
        reg_loss_wd = loss_reg_wd.forward(input=v_z_posterior, target=v_z_prior)
        ## kullback-leibler divergence on \alpha
        reg_loss_kldiv = loss_reg_kldiv.forward(input=v_alpha, target=v_alpha_unif, input_mask=v_x_in_mask)
        if cfg_auto_encoder["loss"]["kldiv"]["enabled"]:
            reg_loss = reg_loss_wd + reg_loss_kldiv
        else:
            reg_loss = reg_loss_wd

        # reconstruction loss(sample-wise mean)
        reconst_loss = loss_reconst.forward(y_ln_prob=v_ln_prob_y, y_true=v_x_out, y_len=x_out_len)

        # total loss
        loss = reconst_loss + reg_loss

    # update model parameters
    if train_mode:
        loss.backward()
        if cfg_optimizer["gradient_clip"] is not None:
            nn.utils.clip_grad_value_(model.parameters(), clip_value=cfg_optimizer["gradient_clip"])
            # DEBUG
            # max_grad = 0.
            # for param in model.parameters():
            #     max_grad = max(max_grad, np.max(np.abs(param.grad.cpu().data.numpy())))
            # print(f"max grad:{max_grad}")
        optimizer.step()

    # compute metrics
    nll = float(reconst_loss)
    nll_token = nll * len(lst_seq_len) / sum(lst_seq_len) # lnq(x|z)*N_sentence/N_token
    metrics = {
        "max_alpha":float(torch.max(v_alpha)),
        "wd":float(reg_loss_wd),
        "kldiv":float(reg_loss_kldiv),
        "nll":nll,
        "nll_token":nll_token,
        "elbo":float(reconst_loss) + float(reg_loss_wd)
    }
    return metrics


def main():

    args = _parse_args()

    # import configuration
    config = importlib.import_module(name=args.config_module)

    cfg_auto_encoder = config.cfg_auto_encoder
    cfg_corpus = config.cfg_corpus
    cfg_optimizer = config.cfg_optimizer

    # show important configurations
    print("corpus:")
    for k, v in cfg_corpus.items():
        print(f"\t{k}:{v}")
    print("optimization:")
    for k, v in cfg_optimizer.items():
        print(f"\t{k}:{v}")

    # instanciate corpora
    corpus = TextLoader(file_path=cfg_corpus["corpus"])
    tokenizer = CharacterTokenizer()
    dictionary = Dictionary.load(file_path=cfg_corpus["dictionary"])

    bos, eos = list(dictionary.special_tokens)
    data_feeder = GeneralSentenceFeeder(corpus = corpus, tokenizer = tokenizer, dictionary = dictionary,
                                        n_minibatch=cfg_optimizer["n_minibatch"], validation_split=0.,
                                        min_seq_len=cfg_corpus["min_seq_len"],
                                        max_seq_len=cfg_corpus["max_seq_len"],
                                        # append `<eos>` at the end of each sentence
                                        bos_symbol=None, eos_symbol=eos)

    # setup logger
    path_log_file = cfg_corpus["log_file_path"]
    if os.path.exists(path_log_file):
        os.remove(path_log_file)
    logger = io.open(path_log_file, mode="w")

    # instanciate variational autoencoder

    ## prior distribution
    n_dim_gmm = cfg_auto_encoder["prior"]["n_dim"]
    n_prior_gmm_component = cfg_auto_encoder["prior"]["n_gmm_component"]

    # calculate l2 norm and stdev of the mean and stdev of prior distribution
    l2_norm, std = calculate_prior_dist_params(expected_swd=cfg_auto_encoder["prior"]["expected_swd"], n_dim_latent=n_dim_gmm)
    vec_alpha = np.full(shape=n_prior_gmm_component, fill_value=1./n_prior_gmm_component)
    mat_mu = generate_random_orthogonal_vectors(n_dim=n_dim_gmm, n_vector=n_prior_gmm_component, l2_norm=l2_norm)
    vec_std = np.ones(shape=n_prior_gmm_component) * std
    prior_distribution = MultiVariateGaussianMixture(vec_alpha=vec_alpha, mat_mu=mat_mu, vec_std=vec_std)
    path_prior = os.path.join(args.save_dir, "prior_distribution.gmm.pickle")
    prior_distribution.save(file_path=path_prior)

    # instanciate variational autoencoder
    cfg_encoder = cfg_auto_encoder["encoder"]
    ## MLP for \alpha, \mu, \sigma
    for param_name in "alpha,mu,sigma".split(","):
        cfg_encoder["lstm"][f"encoder_{param_name}"] = MultiDenseLayer(**cfg_encoder[param_name])
    ## encoder
    encoder = GMMLSTMEncoder(n_vocab=dictionary.max_id+1, device=args.device, **cfg_encoder["lstm"])

    ## sampler(from posterior)
    sampler = GMMSampler(device=args.device, **cfg_auto_encoder["sampler"])

    ## decoder
    decoder = SelfAttentiveLSTMDecoder(device=args.device, **cfg_auto_encoder["decoder"])
    ## prediction layer
    predictor = SimplePredictor(n_dim_out=dictionary.max_id+1, log=True, **cfg_auto_encoder["predictor"])

    ## variational autoencoder
    model = VariationalAutoEncoder(seq_to_gmm_encoder=encoder, gmm_sampler=sampler,
                                   set_to_state_decoder=decoder, state_to_seq_decoder=predictor)
    if len(args.gpus) > 1:
        model = nn.DataParallel(model, device_ids = args.gpus)
    model.to(device=args.device)

    ## loss layers
    loss_wasserstein = EmpiricalSlicedWassersteinDistance(device=args.device, **cfg_auto_encoder["loss"]["empirical_wasserstein"])
    loss_kldiv = MaskedKLDivLoss(scale=cfg_auto_encoder["loss"]["kldiv"]["scale"], reduction="samplewise_mean")
    loss_reconst = PaddedNLLLoss(reduction="samplewise_mean")

    # optimizer
    optimizer = cfg_optimizer["optimizer"](model.parameters(), lr=cfg_optimizer["lr"])

    # start training
    n_epoch = cfg_optimizer["n_epoch"]
    # iterate over epoch
    idx = 0
    for n_epoch in range(n_epoch):
        print(f"epoch:{n_epoch}")

        n_processed = 0
        q = progressbar.ProgressBar(max_value=cfg_corpus["size"])
        q.update(n_processed)

        # iterate over mini-batch
        for train, _ in data_feeder:
            idx += 1
            # training
            train_mode = not(idx % cfg_optimizer["validation_interval"] == 0)
            lst_seq_len, lst_seq = utils.len_pad_sort(lst_seq=train)
            metrics_batch = main_minibatch(model=model, optimizer=optimizer,
                                           prior_distribution=prior_distribution,
                                           loss_reconst=loss_reconst, loss_reg_wd=loss_wasserstein, loss_reg_kldiv=loss_kldiv,
                                           lst_seq=lst_seq, lst_seq_len=lst_seq_len,
                                           device=args.device,
                                           train_mode=train_mode,
                                           cfg_auto_encoder=cfg_auto_encoder,
                                           cfg_optimizer=cfg_optimizer)
            n_processed += len(lst_seq_len)

            # logging and reporting
            metrics = {
                "epoch":n_epoch,
                "processed":n_processed,
                "mode":"train" if train_mode else "val"
            }
            metrics.update(metrics_batch)
            f_value_to_str = lambda v: f"{v:1.7f}" if isinstance(v,float) else f"{v}"
            sep = "\t"
            ## output log file
            if idx == 1:
                s_header = sep.join(metrics.keys()) + "\n"
                logger.write(s_header)
            if args.log_validation_only and train_mode:
                # validation metric only & training mode -> do not output metrics
                pass
            else:
                s_record = sep.join( map(f_value_to_str, metrics.values()) ) + "\n"
                logger.write(s_record)
            logger.flush()

            ## output metrics
            if args.verbose:
                prefix = "train" if train_mode else "val"
                s_print = ", ".join( [f"{prefix}_{k}:{f_value_to_str(v)}" for k,v in metrics.items()] )
                print(s_print)

            # next iteration
            q.update(n_processed)

        # save progress
        path_trained_model_e = os.path.join(args.save_dir, "trained_vae.model." + str(n_epoch))
        print(f"saving...:{path_trained_model_e}")
        torch.save(model.state_dict(), path_trained_model_e)


if __name__ == "__main__":
    main()
    print("finished. good-bye.")
