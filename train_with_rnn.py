#!/usr/bin/env python
# -*- coding:utf-8 -*-

import os, sys, io
from copy import deepcopy
import argparse
from contextlib import ExitStack
import importlib
from typing import List, Dict, Union, Any

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
from model.multi_layer import MultiDenseLayer
from model.encoder import GMMLSTMEncoder
# decoder
from model.attention import  SimpleGlobalAttention
from model.decoder import SelfAttentiveLSTMDecoder
from model.decoder import SimplePredictor
# regularizers
## sampler
from model.noise_layer import GMMSampler
## prior distribution
from distribution.mixture import MultiVariateGaussianMixture
from utility import generate_random_orthogonal_vectors, calculate_prior_dist_params, calculate_mean_l2_between_sample
## loss functions
from model.loss import EmpiricalSlicedWassersteinDistance, GMMSinkhornWassersteinDistance
from model.loss import MaskedKLDivLoss
from model.loss import PaddedNLLLoss
# variational autoencoder
from model.vae import VariationalAutoEncoder
## used for evaluation
from utility import calculate_kldiv

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


def main_minibatch(model, optimizer, prior_distribution, loss_reconst: PaddedNLLLoss,
                   loss_reg_wd: Union[EmpiricalSlicedWassersteinDistance, GMMSinkhornWassersteinDistance],
                   loss_reg_kldiv, lst_seq, lst_seq_len,
                   device, train_mode, cfg_auto_encoder, cfg_optimizer, evaluation_metrics: List[str]) -> Dict[str, float]:

    if train_mode:
        optimizer.zero_grad()

    with ExitStack() as context_stack:
        # if not train mode, enter into no_grad() context
        if not train_mode:
            context_stack.enter_context(torch.no_grad())

        # create encoder input and decoder output(=ground truth)
        ## omit last `<eos>` symbol from the input sequence
        ## x_in = [[i,have,a,pen],[he,like,mary],...]; x_in_len = [4,3,...]
        ## x_out = [[i,have,a,pen,<eos>],[he,like,mary,<eos>],...]; x_out_len = [5,4,...]
        ## 1. encoder input
        x_in = []
        for seq_len, seq in zip(lst_seq_len, lst_seq):
            seq_b = deepcopy(seq)
            del seq_b[seq_len-1]
            x_in.append(seq_b)
        x_in_len = [seq_len - 1 for seq_len in lst_seq_len]
        ## 2. decoder output
        x_out = lst_seq
        x_out_len = lst_seq_len

        # convert to torch.tensor
        ## input
        v_x_in = torch.tensor(x_in, dtype=torch.long).to(device=device)
        v_x_in_len = torch.tensor(x_in_len, dtype=torch.long).to(device=device)
        v_x_in_mask = (v_x_in > 0).float().to(device=device)
        ## output
        v_x_out = torch.tensor(x_out, dtype=torch.long).to(device=device)
        v_x_out_len = torch.tensor(x_out_len, dtype=torch.long).to(device=device)

        ## uniform distribution over $\alpha$
        lst_arr_unif = [np.full(n, 1./n, dtype=np.float32) for n in x_in_len]
        arr_unif = utils.pad_numpy_sequence(lst_arr_unif)
        v_alpha_unif = torch.from_numpy(arr_unif).to(device=device)

        # forward computation of the VAE model
        v_alpha, v_mu, v_sigma, v_z_posterior, v_ln_prob_y, lst_v_alpha, lst_v_mu, lst_v_sigma = \
            model.forward(x_seq=v_x_in, x_seq_len=v_x_in_len, decoder_max_step=max(x_out_len))

        # regularization losses(sample-wise mean)
        ## 1. wasserstein distance between posterior and prior
        if isinstance(loss_reg_wd, EmpiricalSlicedWassersteinDistance):
            ## 1) empirical sliced wasserstein distance
            n_sample = len(x_in) * cfg_auto_encoder["sampler"]["n_sample"]
            v_z_prior = prior_distribution.random(size=n_sample)
            v_z_prior = torch.tensor(v_z_prior, dtype=torch.float32, requires_grad=False).to(device=device)
            v_z_posterior_v = v_z_posterior.view((-1, cfg_auto_encoder["prior"]["n_dim"]))
            reg_loss_wd = loss_reg_wd.forward(input=v_z_posterior_v, target=v_z_prior)
        elif isinstance(loss_reg_wd, GMMSinkhornWassersteinDistance):
            ## 2) sinkhorn wasserstein distance
            v_alpha_prior = torch.tensor(prior_distribution._alpha, dtype=torch.float32, requires_grad=False).to(device=device)
            v_mu_prior = torch.tensor(prior_distribution._mu, dtype=torch.float32, requires_grad=False).to(device=device)
            mat_sigma_prior = np.vstack([np.sqrt(np.diag(cov)) for cov in prior_distribution._cov])
            v_sigma_prior = torch.tensor(mat_sigma_prior, dtype=torch.float32, requires_grad=False).to(device=device)
            reg_loss_wd = loss_reg_wd.forward(lst_vec_alpha_x=lst_v_alpha, lst_mat_mu_x=lst_v_mu, lst_mat_std_x=lst_v_sigma,
                                              vec_alpha_y=v_alpha_prior, mat_mu_y=v_mu_prior, mat_std_y=v_sigma_prior)
        else:
            raise NotImplementedError(f"unsupported regularization layer: {type(loss_reg_wd)}")

        ## 2. (optional) kullback-leibler divergence on \alpha
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
        optimizer.step()

    # compute metrics
    n_sentence = len(x_out)
    n_token = sum(x_out_len)
    nll = float(reconst_loss)
    nll_token = nll * n_sentence / n_token # lnq(x|z)*N_sentence/N_token
    mat_sigma = v_sigma.cpu().data.numpy().flatten()
    mean_sigma = np.mean(mat_sigma[mat_sigma > 0])
    mean_l2_dist = calculate_mean_l2_between_sample(t_z_posterior=v_z_posterior.cpu().data.numpy())
    metrics = {
        "n_sentence":n_sentence,
        "n_token":n_token,
        "mean_max_alpha":float(torch.mean(torch.max(v_alpha, dim=-1)[0])),
        "mean_l2_dist":float(mean_l2_dist),
        "mean_sigma":float(mean_sigma),
        "wd":float(reg_loss_wd),
        "reg_alpha":float(reg_loss_kldiv),
        "nll":nll,
        "nll_token":nll_token,
        "total_cost":float(reconst_loss) + float(reg_loss_wd),
        "kldiv_ana":None,
        "kldiv_mc":None,
        "elbo":None
    }
    if "kldiv_ana" in evaluation_metrics:
        metrics["kldiv_ana"] = calculate_kldiv(lst_v_alpha=lst_v_alpha, lst_v_mu=lst_v_mu, lst_v_sigma=lst_v_sigma,
                                               prior_distribution=prior_distribution, method="analytical") \
                                               * cfg_auto_encoder["sampler"]["n_sample"]
        metrics["elbo"] = metrics["nll"] + metrics["kldiv_ana"]
    if "kldiv_mc" in evaluation_metrics:
        metrics["kldiv_mc"] = calculate_kldiv(lst_v_alpha=lst_v_alpha, lst_v_mu=lst_v_mu, lst_v_sigma=lst_v_sigma,
                                              prior_distribution=prior_distribution, method="monte_carlo", n_mc_sample=1000) \
                                              * cfg_auto_encoder["sampler"]["n_sample"]
        metrics["elbo"] = metrics["nll"] + metrics["kldiv_mc"]

    return metrics


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
        cfg_encoder["lstm"][f"encoder_{param_name}"] = None if cfg_encoder[param_name] is None else MultiDenseLayer(**cfg_encoder[param_name])
    ## encoder
    encoder = GMMLSTMEncoder(n_vocab=dictionary.max_id+1, device=args.device, **cfg_encoder["lstm"])

    ## sampler(from posterior)
    sampler = GMMSampler(device=args.device, **cfg_auto_encoder["sampler"])

    ## decoder
    latent_decoder_name = next(iter(cfg_auto_encoder["decoder"]["latent"]))
    if latent_decoder_name == "simple_attention":
        latent_decoder = SimpleGlobalAttention(**cfg_auto_encoder["decoder"]["latent"][latent_decoder_name])
    elif latent_decoder_name == "multi_head_attention":
        raise NotImplementedError(f"not implemented yet:{latent_decoder_name}")
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
    if is_sliced_wasserstein:
        loss_wasserstein = EmpiricalSlicedWassersteinDistance(device=args.device, **cfg_auto_encoder["loss"]["reg"]["empirical_sliced_wasserstein"])
    else:
        loss_wasserstein = GMMSinkhornWassersteinDistance(device=args.device, **cfg_auto_encoder["loss"]["reg"]["sinkhorn_wasserstein"])
    loss_kldiv = MaskedKLDivLoss(scale=cfg_auto_encoder["loss"]["kldiv"]["scale"], reduction="samplewise_mean")
    loss_reconst = PaddedNLLLoss(reduction="samplewise_mean")

    # optimizer
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
        logger = dict_logger[phase]
        cfg_corpus_t = cfg_corpus[phase]
        q = progressbar.ProgressBar(max_value=cfg_corpus_t["size"])
        n_progress = 0
        q.update(n_progress)
        ## iterate over mini-batch
        for train, _ in dict_data_feeder[phase]:
            n_iteration += 1
            # training
            if cfg_optimizer["validation_interval"] is not None:
                train_mode = not(n_iteration % cfg_optimizer["validation_interval"] == 0)
            else:
                train_mode = True
            lst_seq_len, lst_seq = utils.len_pad_sort(lst_seq=train)
            metrics_batch = main_minibatch(model=model, optimizer=optimizer,
                                           prior_distribution=prior_distribution,
                                           loss_reconst=loss_reconst, loss_reg_wd=loss_wasserstein, loss_reg_kldiv=loss_kldiv,
                                           lst_seq=lst_seq, lst_seq_len=lst_seq_len,
                                           device=args.device,
                                           train_mode=train_mode,
                                           cfg_auto_encoder=cfg_auto_encoder,
                                           cfg_optimizer=cfg_optimizer,
                                           evaluation_metrics=cfg_corpus[phase].get("evaluation_metrics",[]))
            n_processed += len(lst_seq_len)
            n_progress += len(lst_seq_len)

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
            if n_iteration == 1:
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
            q.update(n_progress)

        # save progress
        if args.save_every_epoch:
            path_trained_model_e = os.path.join(args.save_dir, f"lstm_vae.{file_name_suffix}.model." + str(n_epoch))
            print(f"saving...:{path_trained_model_e}")
            torch.save(model.state_dict(), path_trained_model_e)


        #### test phase ####
        if not "test" in dict_data_feeder:
            print("we do not have testset. skip evaluation.")
            continue

        phase = "test"
        print(f"phase:{phase}")
        logger = dict_logger[phase]
        lst_metrics = []
        ## iterate over mini-batch
        for batch, _ in dict_data_feeder[phase]:
            train_mode = False
            lst_seq_len, lst_seq = utils.len_pad_sort(lst_seq=batch)
            metrics_batch = main_minibatch(model=model, optimizer=optimizer,
                                           prior_distribution=prior_distribution,
                                           loss_reconst=loss_reconst, loss_reg_wd=loss_wasserstein, loss_reg_kldiv=loss_kldiv,
                                           lst_seq=lst_seq, lst_seq_len=lst_seq_len,
                                           device=args.device,
                                           train_mode=train_mode,
                                           cfg_auto_encoder=cfg_auto_encoder,
                                           cfg_optimizer=cfg_optimizer,
                                           evaluation_metrics=cfg_corpus[phase].get("evaluation_metrics",[]))
            lst_metrics.append(metrics_batch)

        # logging and reporting
        metrics = {
            "epoch":n_epoch,
            "processed":n_processed,
            "mode":"test"
        }
        vec_n_sentence = np.array([m["n_sentence"] for m in lst_metrics])
        vec_n_token = np.array([m["n_token"] for m in lst_metrics])
        metrics["n_sentence"] = np.sum(vec_n_sentence)
        metrics["n_token"] = np.sum(vec_n_token)
        metrics["n_token_per_sentence"] = metrics["n_token"] / metrics["n_sentence"]
        for metric_name in lst_metrics[0].keys():
            vec_values = np.array([np.nan if m[metric_name] is None else m[metric_name] for m in lst_metrics])
            if metric_name in ["n_sentence","n_token"]:
                continue
            elif metric_name.startswith("mean_"): # sentence-wise mean
                metrics[metric_name] = np.sum(vec_n_sentence * vec_values) / np.sum(vec_n_sentence)
            elif metric_name == "nll_token": # token-wise mean
                metrics[metric_name] = np.sum(vec_n_token * vec_values) / np.sum(vec_n_token)
            else: # token-wise mean * average sentence length
                metrics[metric_name] = np.sum(vec_n_sentence * vec_values) * metrics["n_token_per_sentence"] / metrics["n_token"]

        f_value_to_str = lambda v: f"{v:1.7f}" if isinstance(v,float) else f"{v}"
        sep = "\t"
        ## output log file
        if n_epoch == 0:
            s_header = sep.join(metrics.keys()) + "\n"
            logger.write(s_header)
        s_record = sep.join( map(f_value_to_str, metrics.values()) ) + "\n"
        logger.write(s_record)
        logger.flush()

        ## output metrics
        if args.verbose:
            prefix = "test"
            s_print = ", ".join( [f"{prefix}_{k}:{f_value_to_str(v)}" for k,v in metrics.items()] )
            print(s_print)

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
