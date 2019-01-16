#!/usr/bin/env python
# -*- coding:utf-8 -*-

import os, sys, io
from collections import OrderedDict
from typing import List, Dict, Union, Any, Optional
from copy import deepcopy
from contextlib import ExitStack

import numpy as np

import torch
from torch import nn

from preprocess.dataset_feeder import GeneralSentenceFeeder
from preprocess import utils

# encoders
from model.multi_layer import MultiDenseLayer
from model.encoder import GMMLSTMEncoder
## prior distribution
from distribution.mixture import MultiVariateGaussianMixture
from utility import calculate_mean_l2_between_sample
## loss functions
from model.loss import EmpiricalSlicedWassersteinDistance, GMMSinkhornWassersteinDistance
from model.loss import MaskedKLDivLoss
from model.loss import PaddedNLLLoss
# variational autoencoder
from model.vae import VariationalAutoEncoder
## used for evaluation
from utility import calculate_kldiv

# custom utility
from preprocess import utils


class Estimator(object):

    def __init__(self, model: VariationalAutoEncoder,
                 loss_reconst: PaddedNLLLoss,
                 loss_layer_wd: Union[EmpiricalSlicedWassersteinDistance, GMMSinkhornWassersteinDistance],
                 loss_layer_kldiv: Optional[MaskedKLDivLoss],
                 device,
                 verbose: bool = False):
        self._model = model
        self._loss_reconst = loss_reconst
        self._loss_layer_wd = loss_layer_wd
        self._loss_layer_kldiv = loss_layer_kldiv
        self._device = device
        self._verbose = verbose

    @property
    def reg_wasserstein(self):
        return self._loss_layer_wd

    @property
    def reg_kldiv(self):
        return self._loss_layer_kldiv

    @classmethod
    def load_encoder_from_file(cls, cfg_auto_encoder: Dict[str, Any], n_vocab: int, path_state_dict: str, device):
        # instanciate variational autoencoder
        cfg_encoder = cfg_auto_encoder["encoder"]
        ## MLP for \alpha, \mu, \sigma
        for param_name in "alpha,mu,sigma".split(","):
            cfg_encoder["lstm"][f"encoder_{param_name}"] = None if cfg_encoder[param_name] is None else MultiDenseLayer(**cfg_encoder[param_name])
        ## encoder
        encoder = GMMLSTMEncoder(n_vocab=n_vocab, device=device, **cfg_encoder["lstm"])
        encoder = cls._load_encoder_params(encoder=encoder, path_model_state_dict=path_state_dict, device=device)

        model = VariationalAutoEncoder(seq_to_gmm_encoder=encoder, gmm_sampler=None, set_to_state_decoder=None, state_to_seq_decoder=None)

        obj = cls(model=model, loss_reconst=None, loss_layer_wd=None, loss_layer_kldiv=None, device=device)

        return obj

    @classmethod
    def _load_encoder_params(cls, encoder: nn.Module, path_model_state_dict: str, device):

        encoder_layer_name = "_encoder."

        if device.type == "cpu":
            state_dict=torch.load(path_model_state_dict, map_location="cpu")
        else:
            state_dict=torch.load(path_model_state_dict)

        state_dict_e = OrderedDict()
        for layer_name, params in state_dict.items():
            if layer_name.startswith(encoder_layer_name):
                layer_name_e = layer_name.replace(encoder_layer_name, "")
                state_dict_e[layer_name_e] = params

        encoder.load_state_dict(state_dict=state_dict_e)

        return encoder

    def _detach_computation_graph(self, lst_tensor: List[torch.Tensor]):
        return [tensor.detach() for tensor in lst_tensor]

    def _to_numpy(self, lst_tensor: List[torch.Tensor]):
        return [tensor.cpu().data.numpy() for tensor in lst_tensor]

    def _forward(self, lst_seq, lst_seq_len, optimizer: Optional[torch.optim.Optimizer],
                 prior_distribution: Optional[MultiVariateGaussianMixture],
                 train_mode: bool,
                 inference_mode: bool,
                 clip_gradient: Optional[float] = None):

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
            v_x_in = torch.tensor(x_in, dtype=torch.long, device=self._device)
            v_x_in_len = torch.tensor(x_in_len, dtype=torch.long, device=self._device)
            v_x_in_mask = (v_x_in > 0).float().to(device=self._device)
            ## output
            v_x_out = torch.tensor(x_out, dtype=torch.long, device=self._device)
            v_x_out_len = torch.tensor(x_out_len, dtype=torch.long, device=self._device)

            # if you just want posterior distributions, forward pass will end here.
            if inference_mode:
                lst_v_alpha, lst_v_mu, lst_v_sigma = self._model.infer(x_seq=v_x_in, x_seq_len=v_x_in_len)
                lst_v_alpha = self._detach_computation_graph(lst_v_alpha)
                lst_v_mu = self._detach_computation_graph(lst_v_mu)
                lst_v_sigma = self._detach_computation_graph(lst_v_sigma)
                return lst_v_alpha, lst_v_mu, lst_v_sigma

            # forward computation of the VAE model
            v_alpha, v_mu, v_sigma, v_z_posterior, v_ln_prob_y, lst_v_alpha, lst_v_mu, lst_v_sigma = \
                self._model.forward(x_seq=v_x_in, x_seq_len=v_x_in_len, decoder_max_step=max(x_out_len))

            # regularization losses(sample-wise mean)
            ## 1. wasserstein distance between posterior and prior
            if isinstance(self._loss_layer_wd, EmpiricalSlicedWassersteinDistance):
                ## 1) empirical sliced wasserstein distance
                n_sample = len(x_in) * self._model.sampler_size
                v_z_prior = prior_distribution.random(size=n_sample)
                v_z_prior = torch.tensor(v_z_prior, dtype=torch.float32, requires_grad=False).to(device=self._device)
                v_z_posterior_v = v_z_posterior.view((-1, prior_distribution.n_dim))
                reg_loss_wd = self._loss_layer_wd.forward(input=v_z_posterior_v, target=v_z_prior)
            elif isinstance(self._loss_layer_wd, GMMSinkhornWassersteinDistance):
                ## 2) (marginalized) sinkhorn wasserstein distance
                v_alpha_prior = torch.tensor(prior_distribution._alpha, dtype=torch.float32, requires_grad=False).to(device=self._device)
                v_mu_prior = torch.tensor(prior_distribution._mu, dtype=torch.float32, requires_grad=False).to(device=self._device)
                mat_sigma_prior = np.vstack([np.sqrt(np.diag(cov)) for cov in prior_distribution._cov])
                v_sigma_prior = torch.tensor(mat_sigma_prior, dtype=torch.float32, requires_grad=False).to(device=self._device)
                reg_loss_wd = self._loss_layer_wd.forward(lst_vec_alpha_x=lst_v_alpha, lst_mat_mu_x=lst_v_mu, lst_mat_std_x=lst_v_sigma,
                                                    vec_alpha_y=v_alpha_prior, mat_mu_y=v_mu_prior, mat_std_y=v_sigma_prior)
            else:
                raise NotImplementedError(f"unsupported regularization layer: {type(self._loss_layer_wd)}")

            ## 2. (optional) kullback-leibler divergence on \alpha
            if self._loss_layer_kldiv is not None:
                ## uniform distribution over $\alpha$
                lst_arr_unif = [np.full(n, 1./n, dtype=np.float32) for n in x_in_len]
                arr_unif = utils.pad_numpy_sequence(lst_arr_unif)
                v_alpha_unif = torch.from_numpy(arr_unif).to(device=self._device)
                reg_loss_kldiv = self._loss_layer_kldiv.forward(input=v_alpha, target=v_alpha_unif, input_mask=v_x_in_mask)
                reg_loss = reg_loss_wd + reg_loss_kldiv
            else:
                reg_loss_kldiv = None
                reg_loss = reg_loss_wd

            # reconstruction loss(sample-wise mean)
            reconst_loss = self._loss_reconst.forward(y_ln_prob=v_ln_prob_y, y_true=v_x_out, y_len=x_out_len)

            # total loss
            loss = reconst_loss + reg_loss

        # update model parameters
        if train_mode:
            loss.backward()
            if clip_gradient is not None:
                nn.utils.clip_grad_value_(self._model.parameters(), clip_value=clip_gradient)
            optimizer.step()

        return reconst_loss, reg_loss_wd, reg_loss_kldiv, lst_v_alpha, lst_v_mu, lst_v_sigma, v_z_posterior


    def _compute_metrics_minibatch(self, n_sentence, n_token,
                         reconst_loss: torch.Tensor, reg_loss_wd: torch.Tensor, reg_loss_kldiv: Optional[torch.Tensor],
                         lst_v_alpha: List[torch.Tensor], lst_v_mu: List[torch.Tensor], lst_v_sigma: List[torch.Tensor],
                         v_z_posterior: torch.Tensor,
                         prior_distribution: MultiVariateGaussianMixture,
                         evaluation_metrics: List[str]):
        # compute metrics for single minibatch
        nll = float(reconst_loss)
        nll_token = nll * n_sentence / n_token # lnq(x|z)*N_sentence/N_token
        mat_sigma = torch.cat(lst_v_sigma).cpu().data.numpy().flatten()
        mean_sigma = np.mean(mat_sigma)
        mean_l2_dist = calculate_mean_l2_between_sample(t_z_posterior=v_z_posterior.cpu().data.numpy())
        mean_max_alpha = np.mean([np.max(v_alpha.cpu().data.numpy()) for v_alpha in lst_v_alpha])
        metrics = {
            "n_sentence":n_sentence,
            "n_token":n_token,
            "mean_max_alpha":mean_max_alpha,
            "mean_l2_dist":float(mean_l2_dist),
            "mean_sigma":float(mean_sigma),
            "wd":float(reg_loss_wd),
            "wd_scale":self._loss_layer_wd.scale,
            "reg_alpha":np.nan if reg_loss_kldiv is None else float(reg_loss_kldiv),
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
                                                   * self._model.sampler_size
            metrics["elbo"] = metrics["nll"] + metrics["kldiv_ana"]
        if "kldiv_mc" in evaluation_metrics:
            metrics["kldiv_mc"] = calculate_kldiv(lst_v_alpha=lst_v_alpha, lst_v_mu=lst_v_mu, lst_v_sigma=lst_v_sigma,
                                                  prior_distribution=prior_distribution, method="monte_carlo", n_mc_sample=1000) \
                                                  * self._model.sampler_size
            metrics["elbo"] = metrics["nll"] + metrics["kldiv_mc"]

        return metrics


    def train_single_step(self, lst_seq, lst_seq_len, optimizer, prior_distribution, clip_gradient_value: Optional[float] = None,
                          evaluation_metrics: Optional[List[str]] = None):

        reconst_loss, reg_loss_wd, reg_loss_kldiv, lst_v_alpha, lst_v_mu, lst_v_sigma, v_z_posterior = \
        self._forward(lst_seq=lst_seq, lst_seq_len=lst_seq_len, optimizer=optimizer, prior_distribution=prior_distribution,
                      train_mode=True, inference_mode=False,
                      clip_gradient=clip_gradient_value)

        n_sentence = len(lst_seq)
        n_token = sum(lst_seq_len)

        if evaluation_metrics is not None:
            dict_metrics = self._compute_metrics_minibatch(
                n_sentence=n_sentence, n_token=n_token,
                reconst_loss=reconst_loss, reg_loss_wd=reg_loss_wd, reg_loss_kldiv=reg_loss_kldiv,
                lst_v_alpha=lst_v_alpha, lst_v_mu=lst_v_mu, lst_v_sigma=lst_v_sigma,
                v_z_posterior=v_z_posterior,
                prior_distribution=prior_distribution,
                evaluation_metrics=evaluation_metrics
            )
        else:
            dict_metrics = {}

        return dict_metrics


    def test_single_step(self, lst_seq, lst_seq_len, prior_distribution, evaluation_metrics: [List[str]]):

        reconst_loss, reg_loss_wd, reg_loss_kldiv, lst_v_alpha, lst_v_mu, lst_v_sigma, v_z_posterior = \
        self._forward(lst_seq=lst_seq, lst_seq_len=lst_seq_len, optimizer=None, prior_distribution=prior_distribution,
                      train_mode=False, inference_mode=False, clip_gradient=None)

        n_sentence = len(lst_seq)
        n_token = sum(lst_seq_len)

        dict_metrics = self._compute_metrics_minibatch(
            n_sentence=n_sentence, n_token=n_token,
            reconst_loss=reconst_loss, reg_loss_wd=reg_loss_wd, reg_loss_kldiv=reg_loss_kldiv,
            lst_v_alpha=lst_v_alpha, lst_v_mu=lst_v_mu, lst_v_sigma=lst_v_sigma,
            v_z_posterior=v_z_posterior,
            prior_distribution=prior_distribution,
            evaluation_metrics=evaluation_metrics
        )

        return dict_metrics


    def inference_single_step(self, lst_seq, lst_seq_len, return_numpy: Optional[bool] = True):

        # tup_lst_params = lst_v_alpha, lst_v_mu, lst_v_sigma
        tup_lst_params = \
                self._forward(lst_seq=lst_seq, lst_seq_len=lst_seq_len, optimizer=None, prior_distribution=None,
                              train_mode=False, inference_mode=True, clip_gradient=None)

        if not return_numpy:
            return tup_lst_params
        else:
            return tuple(self._to_numpy(lst_tensor=lst_params) for lst_params in tup_lst_params)

    def train_prior_distribution(self,
                    cfg_optimizer: Dict[str, Any],
                    cfg_sinkhorn_wasserstein: Optional[Dict[str, Any]],
                    prior_distribution: MultiVariateGaussianMixture,
                    data_feeder: GeneralSentenceFeeder):

        # instanciate wasserstein distance layer
        if cfg_sinkhorn_wasserstein is None:
            if isinstance(self._loss_layer_wd, EmpiricalSlicedWassersteinDistance):
                raise AttributeError(f"{self._loss_layer_wd.__class__.__name__} is not supported.")
            if self._verbose:
                print("prior distribution updater will reuse vae's wasserstein loss layer.")
            loss_layer_wd = self._loss_layer_wd
        else:
            loss_layer_wd = GMMSinkhornWassersteinDistance(device=self._device, **cfg_sinkhorn_wasserstein)

        # create parameters for prior distribution
        v_alpha_y = torch.tensor(prior_distribution._alpha, device=self._device, dtype=torch.float32, requires_grad=False)
        v_mu_y = torch.tensor(prior_distribution._mu, device=self._device, dtype=torch.float32, requires_grad=True)
        mat_sigma_y = np.vstack([np.sqrt(np.diag(cov)) for cov in prior_distribution._cov])
        v_sigma_y_ln = torch.tensor(np.log(mat_sigma_y), device=self._device, dtype=torch.float32, requires_grad=True)

        optimizer = cfg_optimizer["optimizer"](params=[v_mu_y, v_sigma_y_ln], lr=cfg_optimizer["lr"])

        for idx, (batch, _) in enumerate(data_feeder):
            optimizer.zero_grad()

            v_sigma_y = torch.exp(v_sigma_y_ln)

            lst_seq_len, lst_seq = utils.len_pad_sort(lst_seq=batch)
            lst_v_alpha, lst_v_mu, lst_v_sigma = self.inference_single_step(lst_seq=lst_seq, lst_seq_len=lst_seq_len, return_numpy=False)

            wd = loss_layer_wd.forward(lst_v_alpha, lst_v_mu, lst_v_sigma, v_alpha_y, v_mu_y, v_sigma_y)

            wd.backward()
            optimizer.step()

            if self._verbose and idx == 0:
                print("wd before updating prior:", wd.item())

        if self._verbose:
            print("wd after updating prior:", wd.item())

        # reconstruct prior distribution
        x_alpha = v_alpha_y.cpu().data.numpy()
        x_mu = v_mu_y.cpu().data.numpy()
        x_sigma = np.exp(v_sigma_y_ln.cpu().data.numpy())

        prior_updated = MultiVariateGaussianMixture(vec_alpha=x_alpha, mat_mu=x_mu, mat_cov = x_sigma**2)

        return prior_updated