#!/usr/bin/env python
# -*- coding:utf-8 -*-

import sys, io, os
import numpy as np
from pathos import multiprocessing
from typing import List, Tuple, Union, Iterator, Callable, Optional
import torch
from torch.nn.modules.loss import _Loss
from torch.nn import functional as F

__ROOT_DIR = os.path.join( os.path.dirname(__file__), "../")
sys.path.append(__ROOT_DIR)

from distribution.mixture import MultiVariateGaussianMixture, UniVariateGaussianMixture


class BaseAnnealableLoss(_Loss):

    def __init__(self, scale: Union[int, float, Callable[[int], float]], size_average=None, reduce=None, reduction='samplewise_mean'):

        super(BaseAnnealableLoss, self).__init__(size_average, reduce, reduction)

        self._scale_func = scale
        self.update_scale_parameter(n_processed=0)

    @property
    def scale(self):
        return self._scale

    @scale.setter
    def scale(self, value):
        self._scale = value

    def update_scale_parameter(self, n_processed: int):
        """

        :param n_processed: number of processed sentences so far.
        :return: scale value
        """
        if isinstance(self._scale_func, (int, float)):
            self._scale = self._scale_func
        elif isinstance(self._scale_func, Callable):
            self._scale = self._scale_func(n_processed)
        else:
            raise NotImplementedError("unsupported scale parameter type. it must be numeric or callable that returns numeric value.")


class PaddedNLLLoss(_Loss):

    def _elementwise_mean(self, y_ln_prob: torch.Tensor, y_true: torch.Tensor, y_len: Union[List[int], torch.Tensor]):

        y_ln_prob_seq = torch.cat(tuple(y_ln_prob[b,:seq_len,:] for b, seq_len in enumerate(y_len)), dim=0)

        if y_true.ndimension() == 1:
            y_true_seq = y_true
        elif y_true.ndimension() == 2:
            y_true_seq = torch.cat(tuple(y_true[b,:seq_len] for b, seq_len in enumerate(y_len)), dim=0)
        else:
            raise AttributeError("unsupported input dimension.")
        loss = F.nll_loss(input=y_ln_prob_seq, target=y_true_seq)
        return loss

    def _samplewise_mean(self, y_ln_prob: torch.Tensor, y_true: torch.Tensor, y_len: Union[List[int], torch.Tensor]):

        assert y_true.ndimension() == 2, "`samplewise_mean` requires two-dimensional `y_true` tensor."

        n_mb = len(y_len)
        loss = None
        for b, seq_len in enumerate(y_len):
            y_ln_b = y_ln_prob[b,:seq_len,:]
            y_true_b = y_true[b,:seq_len]
            loss_b = F.nll_loss(input=y_ln_b, target=y_true_b, reduction="sum")
            if loss is None:
                loss = loss_b
            else:
                loss = loss + loss_b
        loss = torch.div(loss, n_mb)
        return loss

    def forward(self, y_ln_prob: torch.Tensor, y_true: torch.Tensor, y_len: Union[List[int], torch.Tensor]):

        if self.reduction == "elementwise_mean":
            loss = self._elementwise_mean(y_ln_prob, y_true, y_len)
        elif self.reduction == "samplewise_mean":
            loss = self._samplewise_mean(y_ln_prob, y_true, y_len)
        else:
            raise NotImplementedError("unsupported reduction method: %s" % self.reduction)

        return loss


class MaskedKLDivLoss(_Loss):

    __EPS = 1E-5

    def __init__(self, scale: float = 1.0, size_average=None, reduce=None, reduction='samplewise_mean'):

        super(MaskedKLDivLoss, self).__init__(size_average, reduce, reduction)

        self._scale = scale


    def forward(self, input: torch.Tensor, target: torch.Tensor, input_mask: torch.Tensor):

        if input_mask.is_floating_point():
            input_mask = input_mask.float()
        # n_elem = torch.sum(input_mask, dim=-1)
        # batch_loss = torch.sum(input_mask * input * (torch.log(input + self.__EPS) - torch.log(target + self.__EPS)), dim=-1) / n_elem
        batch_loss = torch.sum(input_mask * input * (torch.log(input + self.__EPS) - torch.log(target + self.__EPS)), dim=-1)

        if self.reduction == "samplewise_mean":
            loss = torch.mean(batch_loss)
        elif self.reduction == "sum":
            loss = torch.sum(batch_loss)
        loss = loss * self._scale

        return loss


class EmpiricalSlicedWassersteinDistance(BaseAnnealableLoss):

    def __init__(self, n_slice, scale=1.0, size_average=None, reduce=None, reduction='elementwise_mean', device=torch.device("cpu")):
        """
        sliced wasserstein distance using samples from source and target distributions

        :param n_slice: number of slices. i.e. number of radon transformation
        :param scale: scale parameter. output will be scale * distance
        :param reduction: it must be `elementwise_mean`
        """
        super(EmpiricalSlicedWassersteinDistance, self).__init__(scale, size_average, reduce, reduction)

        self._n_slice = n_slice
        self._device = device

    def _sample_circular(self, n_dim, size=None, requires_grad=False):
        if size is None:
            v = np.random.normal(size=n_dim).astype(np.float32)
            v /= np.sqrt(np.sum(v**2))
        else:
            v = np.random.normal(size=n_dim*size).astype(np.float32).reshape((size, n_dim))
            v /= np.linalg.norm(v, axis=1, ord=2).reshape((-1,1))

        t = torch.tensor(data=v, dtype=torch.float, requires_grad=requires_grad, device=self._device)

        return t


    def forward(self, input, target):

        n_mb, n_dim = input.shape

        # sliced wesserstein distance
        loss = torch.tensor(0., dtype=torch.float, requires_grad=True, device=self._device)
        for t in range(self._n_slice):
            t_theta = self._sample_circular(n_dim=n_dim)
            x_t,_ = torch.matmul(input, t_theta).topk(k=n_mb)
            y_t,_ = torch.matmul(target, t_theta).topk(k=n_mb)

            loss = torch.add(loss, torch.mean(torch.pow(x_t-y_t,2)))

        # it returns sample-wise mean
        loss = torch.div(loss, self._n_slice) * self._scale

        return loss


class GMMSlicedWassersteinDistance(object):

    __dtype = np.float32

    def __init__(self, n_dim: int, n_slice: int, n_integral_point: int, inv_cdf_method: str, exponent: int = 2, scale_gradient=True, **kwargs):
        lst_accept = ["analytical","empirical"]
        assert inv_cdf_method in lst_accept, "argument `inv_cdf_method` must be one of these: %s" % "/".join(lst_accept)

        self._n_dim = n_dim
        self._n_slice = n_slice
        self._n_integral = n_integral_point
        self._inv_cdf_method = inv_cdf_method
        self._exponent = exponent
        self._scale_gradient = scale_gradient

        self._integration_point = np.arange(self._n_integral)/self._n_integral + 1. / (2*self._n_integral)
        self._init_extend(**kwargs)

    def _init_extend(self, **kwargs):
        pass

    def _init_grad(self, seq_len):
        g_alpha = np.zeros(seq_len, dtype=self.__dtype)
        g_mu = np.zeros((seq_len, self._n_dim), dtype=self.__dtype)
        g_sigma = np.zeros(seq_len, dtype=self.__dtype)

        return g_alpha, g_mu, g_sigma

    def _sample_circular_distribution(self, size: int):
        v = np.random.normal(size=self._n_dim * size).astype(self.__dtype).reshape((size, self._n_dim))
        v /= np.linalg.norm(v, axis=1, ord=2).reshape((-1,1))
        return v

    def _grad_wasserstein1d(self, p_x: UniVariateGaussianMixture, p_y: UniVariateGaussianMixture) -> (float, np.ndarray, np.ndarray, np.ndarray):
        vec_tau = self._integration_point
        n_integral = vec_tau.size
        inv_cdf_method = self._inv_cdf_method
        exponent = self._exponent

        if inv_cdf_method == "analytical":
            t_x = p_x.inv_cdf(vec_tau)
            t_y = p_y.inv_cdf (vec_tau)
        elif inv_cdf_method == "empirical":
            t_x = p_x.inv_cdf_empirical(vec_tau, n_approx=3*n_integral)
            t_y = p_y.inv_cdf_empirical(vec_tau, n_approx=3*n_integral)
        else:
            raise AssertionError("never happen.")

        # calculate wasserstein distance
        dist = np.mean(np.power(t_x - t_y, exponent))

        # calculate gradients
        vec_grad_alpha = np.zeros_like(p_x._alpha, dtype=self.__dtype)
        vec_grad_mu = np.zeros_like(p_x._mu, dtype=self.__dtype)
        vec_grad_sigma = np.zeros_like(p_x._std, dtype=self.__dtype)
        pdf_x = p_x.pdf(t_x)
        for k in range(p_x._n_k):
            # grad_alpha = \int (F_x_inv - F_y_inv)*F_x_k/P_x
            cdf_x_k = p_x.cdf_component(u=t_x, k=k) / p_x._alpha[k]
            grad_alpha_k = 2*np.mean( (t_x - t_y)*cdf_x_k/pdf_x)

            # grad_mu = \int (F_x_inv - F_y_inv)*P_x_k/P_x
            pdf_x_k = p_x.pdf_component(u=t_x, k=k)
            grad_mu_k = 2*np.mean( (t_x - t_y)*pdf_x_k/pdf_x )

            # grad_sigma = \int (F_x_inv - F_y_inv)*z_k*P_x_k/P_x
            z_k = (t_x - p_x._mu[k])/p_x._std[k]
            grad_sigma_k = 2*np.mean( (t_x - t_y)*z_k*pdf_x_k/pdf_x )

            vec_grad_alpha[k] = grad_alpha_k
            vec_grad_mu[k] = grad_mu_k
            vec_grad_sigma[k] = grad_sigma_k

        # auto scaling
        if self._scale_gradient:
            s_mu = np.mean(np.abs(vec_grad_mu))
            s_alpha = np.mean(np.abs(vec_grad_alpha))
            vec_grad_alpha *= (s_mu / s_alpha)

        return dist, vec_grad_alpha, vec_grad_mu, vec_grad_sigma


    def sliced_wasserstein_distance_single(self, f_x: MultiVariateGaussianMixture, f_y: MultiVariateGaussianMixture, mat_theta: np.ndarray = None) -> (float, np.ndarray, np.ndarray, np.ndarray):

        # initialize
        n_slice = self._n_slice
        dist = 0.
        seq_len = f_x.n_component
        g_alpha, g_mu, g_sigma = self._init_grad(seq_len=seq_len)

        # sample mapping hyperplane
        if mat_theta is None:
            mat_theta = self._sample_circular_distribution(size=n_slice)
        else:
            mat_theta = mat_theta
        # map multi-dimensional distribution into each hyperplane
        lst_p_x = [f_x.radon_transform(vec_theta=theta) for theta in mat_theta]
        lst_p_y = [f_y.radon_transform(vec_theta=theta) for theta in mat_theta]

        # calculate distance in each sliced dimension
        grad_func = lambda p_x, p_y: self._grad_wasserstein1d(p_x=p_x, p_y=p_y)
        iter_grad = map(grad_func, lst_p_x, lst_p_y)

        # take average for disance and gradients
        for theta, (dist_t, g_alpha_t, g_mu_t, g_sigma_t) in zip(mat_theta, iter_grad):
            dist += dist_t
            g_alpha += g_alpha_t
            g_mu += np.expand_dims(g_mu_t, axis=1) * theta
            g_sigma += g_sigma_t

        dist /= n_slice
        g_alpha /= n_slice
        g_mu /= n_slice
        g_sigma /= n_slice

        return dist, g_alpha, g_mu, g_sigma


    def sliced_wasserstein_distance_batch(self,
                                          lst_gmm_x: Iterator[Tuple[np.ndarray, np.ndarray, np.ndarray]],
                                          lst_gmm_y: Iterator[Tuple[np.ndarray, np.ndarray, np.ndarray]]) \
                                            -> (List[float], List[np.ndarray], List[np.ndarray], List[np.ndarray]):

        lst_f_x = [MultiVariateGaussianMixture(vec_alpha=alpha, mat_mu=mu, vec_std=sigma) for alpha, mu, sigma in lst_gmm_x]
        lst_f_y = [MultiVariateGaussianMixture(vec_alpha=alpha, mat_mu=mu, vec_std=sigma) for alpha, mu, sigma in lst_gmm_y]

        mat_theta = self._sample_circular_distribution(size=self._n_slice)

        grad_func = lambda f_x, f_y: self.sliced_wasserstein_distance_single(f_x, f_y, mat_theta)
        iter_grad = map(grad_func, lst_f_x, lst_f_y)

        return tuple(map(list, zip(*iter_grad)))



class GMMSlicedWassersteinDistance_Parallel(GMMSlicedWassersteinDistance):

    __dtype = np.float32
    __num_cpu = multiprocessing.cpu_count()

    def sliced_wasserstein_distance_batch(self,
                                          lst_gmm_x: Iterator[Tuple[np.ndarray, np.ndarray, np.ndarray]],
                                          lst_gmm_y: Iterator[Tuple[np.ndarray, np.ndarray, np.ndarray]]) \
                                            -> (List[float], List[np.ndarray], List[np.ndarray], List[np.ndarray]):

        _pool = multiprocessing.Pool(processes=self.__num_cpu)

        lst_f_x = [MultiVariateGaussianMixture(vec_alpha=alpha, mat_mu=mu, vec_std=sigma) for alpha, mu, sigma in lst_gmm_x]
        lst_f_y = [MultiVariateGaussianMixture(vec_alpha=alpha, mat_mu=mu, vec_std=sigma) for alpha, mu, sigma in lst_gmm_y]
        lst_f = zip(lst_f_x, lst_f_y)

        mat_theta = self._sample_circular_distribution(size=self._n_slice)

        grad_func = lambda f_x_and_y: self.sliced_wasserstein_distance_single(*(f_x_and_y), mat_theta=mat_theta)

        n_batch = len(lst_f_x)
        chunksize = n_batch // self.__num_cpu
        iter_grad = _pool.imap(grad_func, lst_f, chunksize=chunksize)
        obj_ret = tuple(map(list, zip(*iter_grad)))

        _pool.close()

        return obj_ret


class GMMSinkhornWassersteinDistance(BaseAnnealableLoss):

    def __init__(self, marginalize_posterior: bool,
                 scale=1.0, sinkhorn_lambda=1.0, sinkhorn_iter_max=100, sinkhorn_threshold=0.1,
                 weight_function_for_sequence_length: Optional[Callable] = None,
                 size_average=None, reduce=None, reduction='samplewise_mean', device=torch.device("cpu")):
        """
        approximated wasserstein distance between two gaussian mixtures using sinkhorn algorithm.

        :param marginalize_posterior: marginalize out posterior distribution(=E_x[p(z|x]) or not
        :param scale: scale parameter. output will be scale * wasserstein distance
        :param sinkhorn_lambda: smoothing parameter of sinkhorn algorithm. it will be multiplied to distance matrix.
        :param sinkhorn_iter_max: maximum number of iterations of sinkhorn algorithm.
        :param sinkhorn_threshold: threshold that is used to detect convergence of sinkhorn algorithm.
        :param reduction: it must be `samplewise_mean`
        """

        assert reduction == "samplewise_mean", "this metric supports sample-wise mean only."

        super(GMMSinkhornWassersteinDistance, self).__init__(scale, size_average, reduce, reduction)

        self._marginalize_posterior = marginalize_posterior
        self._lambda = sinkhorn_lambda
        self._n_iter_max = sinkhorn_iter_max
        self._threshold = sinkhorn_threshold
        self._weight_function = weight_function_for_sequence_length
        self._device = device

    def _generate_weight(self, lst_seq_len: List[int]):
        n_mb = len(lst_seq_len)
        if self._weight_function is None:
            return np.full(shape=n_mb, fill_value=1./n_mb)
        else:
            vec_w = np.array([self._weight_function(seq_len) for seq_len in lst_seq_len])
            vec_w = vec_w / np.sum(vec_w)
            return vec_w

    def _matrix_dist_l2_sq(self, mat_x: torch.Tensor, mat_y: torch.Tensor):
        ret = torch.sum(mat_x**2, dim=-1, keepdim=True) \
                   + torch.transpose(torch.sum(mat_y**2, dim=-1, keepdim=True),0,1) \
                   - 2*torch.mm(mat_x, torch.transpose(mat_y,0,1))
        return ret

    def _calculate_distance_matrix(self, v_mat_mu_x: torch.Tensor, v_mat_std_x: torch.Tensor, v_mat_mu_y: torch.Tensor, v_mat_std_y: torch.Tensor):

        n_dim = v_mat_mu_x.shape[-1]
        # l2_sq_mu = (n_c_x, n_c_y)
        l2_sq_mu = self._matrix_dist_l2_sq(v_mat_mu_x, v_mat_mu_y)
        # tr_sigma = (n_c_x, n_c_y)
        if v_mat_std_x.shape[-1] == n_dim:
            v_mat_std_x_ = v_mat_std_x
        else:
            v_mat_std_x_ = v_mat_std_x.repeat(1, n_dim)
        tr_sigma = self._matrix_dist_l2_sq(v_mat_std_x_, v_mat_std_y)

        return l2_sq_mu + tr_sigma

    def _sinkhorn_algorithm(self, vec_p: torch.Tensor, vec_q: torch.Tensor, mat_dist: torch.Tensor):

        vec_ln_p = torch.log(vec_p)
        vec_ln_q = torch.log(vec_q)
        vec_ln_a = torch.zeros_like(vec_p, dtype=torch.float, device=self._device)
        vec_ln_b = torch.zeros_like(vec_q, dtype=torch.float, device=self._device)
        mat_ln_k = -mat_dist * self._lambda

        for n_iter in range(self._n_iter_max):

            vec_ln_a_prev = vec_ln_a.detach().clone()
            vec_ln_a = vec_ln_p - torch.logsumexp(mat_ln_k + vec_ln_b.view(1,-1), dim=-1)
            vec_ln_b = vec_ln_q - torch.logsumexp(mat_ln_k + vec_ln_a.view(-1,1), dim=0)

            # termination
            ## difference with condition
            err = (vec_ln_a.detach() - vec_ln_a_prev).abs().sum()
            if err < self._threshold:
                break

        mat_gamma = torch.exp(vec_ln_a.view(-1,1) + mat_ln_k + vec_ln_b.view(1,-1))
        dist = torch.sum(mat_gamma*mat_dist)

        return dist, mat_gamma

    def _marginalize_gaussian_mixture(self, lst_vec_alpha: List[torch.Tensor], lst_mat_mu: List[torch.Tensor], lst_mat_std: List[torch.Tensor],
                                      lst_weight: List[float]):

        # \alpha = \sum_{b}{w_b * \alpha_b}: (\sum{N_component})
        lst_vec_alpha_w = [vec_alpha * w for vec_alpha, w in zip(lst_vec_alpha, lst_weight)]
        vec_alpha = torch.cat(lst_vec_alpha_w)
        # \mu: (\sum{N_component}, N_dim)
        mat_mu = torch.cat(lst_mat_mu, dim=0)
        # \sigma: (\sum{N_component}, N_dim) or (\sum{N_component}, 1)
        mat_std = torch.cat(lst_mat_std, dim=0)

        return vec_alpha, mat_mu, mat_std

    def forward(self, lst_vec_alpha_x: List[torch.Tensor], lst_mat_mu_x: List[torch.Tensor], lst_mat_std_x: List[torch.Tensor],
                vec_alpha_y: torch.Tensor, mat_mu_y: torch.Tensor, mat_std_y: torch.Tensor):
        """
        calculate approximated wasserstein distance between posterior gaussian mixtures and prior gaussian mixture using sinkhorn algorithm.

        :param lst_vec_alpha_x: posterior. list of the packed scale vectors.
        :param lst_mat_mu_x: posterior. list of the sequence of packed mean vectors
        :param lst_mat_std_x: posterior. list of the sequence of packed standard deviation vectors(=sqrt of the diagonal elements of covariance matrix)
        :return: mean of the square of approximate 2-wasserstein distance * scale parameter
        """

        lst_seq_len = [len(vec_alpha) for vec_alpha in lst_vec_alpha_x]
        lst_weight = self._generate_weight(lst_seq_len=lst_seq_len)

        if self._marginalize_posterior:
            vec_alpha_x, mat_mu_x, mat_std_x = self._marginalize_gaussian_mixture(lst_vec_alpha_x, lst_mat_mu_x, lst_mat_std_x, lst_weight)
            v_mat_dist = self._calculate_distance_matrix(v_mat_mu_x=mat_mu_x, v_mat_std_x=mat_std_x, v_mat_mu_y=mat_mu_y, v_mat_std_y=mat_std_y)
            v_wd_sq, mat_gamma = self._sinkhorn_algorithm(vec_p=vec_alpha_x, vec_q=vec_alpha_y, mat_dist=v_mat_dist)
            v_wd_sq = v_wd_sq * self._scale

            return v_wd_sq

        else:
            weight_sum = 0
            v_wd_sq = torch.zeros(1, dtype=torch.float, requires_grad=True, device=self._device)
            for v_alpha, v_mu, v_std, weight in zip(lst_vec_alpha_x, lst_mat_mu_x, lst_mat_std_x, lst_weight):
                v_mat_dist = self._calculate_distance_matrix(v_mat_mu_x=v_mu, v_mat_std_x=v_std, v_mat_mu_y=mat_mu_y, v_mat_std_y=mat_std_y)
                v_wd_sq_b, mat_gamma = self._sinkhorn_algorithm(vec_p=v_alpha, vec_q=vec_alpha_y, mat_dist=v_mat_dist)
                if not torch.isnan(v_wd_sq_b):
                    v_wd_sq = v_wd_sq + v_wd_sq_b * weight
                    weight_sum += weight

            if weight_sum > 0:
                v_wd_sq = torch.div(v_wd_sq, weight_sum) * self._scale

            return v_wd_sq


class GMMApproxKLDivergence(BaseAnnealableLoss):

    def __init__(self, marginalize_posterior: bool = False,
                 scale=1.0,
                 multiplier=1.0,
                 weight_function_for_sequence_length: Optional[Callable] = None,
                 size_average=None, reduce=None, reduction='samplewise_mean', device=torch.device("cpu")):
        """
        approximated kullback-leibler divergence between two gaussian mixtures using variational approximation.

        :param marginalize_posterior: marginalize out posterior distribution(=E_x[p(z|x]) or not
        :param scale: scale parameter. output will be scale * wasserstein distance
        :param multiplier: multiplication value over the output. it should be the number of samples sampled from posterior distribution.
        :param weight_function_for_sequence_length: another scale parameter that depends on the sequence length.
        :param reduction: it must be `samplewise_mean`
        """

        # assert marginalize_posterior == False, "this metric does not support `marginalize_posterior` option."

        assert reduction == "samplewise_mean", "this metric supports sample-wise mean only."

        super(GMMApproxKLDivergence, self).__init__(scale, size_average, reduce, reduction)

        self._marginalize_posterior = marginalize_posterior
        self._weight_function = weight_function_for_sequence_length
        self._multiplier = float(multiplier)
        self._device = device

    def _generate_weight(self, lst_seq_len: List[int]):
        n_mb = len(lst_seq_len)
        if self._weight_function is None:
            return np.full(shape=n_mb, fill_value=1./n_mb)
        else:
            vec_w = np.array([self._weight_function(seq_len) for seq_len in lst_seq_len])
            vec_w = vec_w / np.sum(vec_w)
            return vec_w

    def _kldiv_diag_parallel(self, mat_mu_x:torch.Tensor, mat_cov_x:torch.Tensor,
                             mat_mu_y: Optional[torch.Tensor] = None, mat_cov_y: Optional[torch.Tensor] = None):
        n_dim = mat_mu_x.shape[0]

        # return: (n_c_x, n_c_y)
        # return[i,j] = KL(p_x_i||p_y_j)

        if mat_mu_y is None:
            mat_mu_y = mat_mu_x
        if mat_cov_y is None:
            mat_cov_y = mat_cov_x

        v_ln_nu_sum_x = torch.sum(torch.log(mat_cov_x), dim=-1)
        v_ln_nu_sum_y = torch.sum(torch.log(mat_cov_y), dim=-1)
        v_det_term = v_ln_nu_sum_x.reshape(-1,1) + v_ln_nu_sum_y.reshape(1,-1)

        v_tr_term = torch.mm(mat_cov_x, torch.transpose(1./mat_cov_y, 0, 1))

        v_quad_term_xx = torch.mm(mat_mu_x**2, torch.transpose(1./mat_cov_y, 0, 1))
        v_quad_term_xy = torch.mm(mat_mu_x, torch.transpose(mat_mu_y/mat_cov_y, 0, 1))
        v_quad_term_yy = torch.sum(mat_mu_y**2 / mat_cov_y, dim=-1)
        v_quad_term = v_quad_term_xx - 2*v_quad_term_xy + v_quad_term_yy.reshape(1,-1)

        mat_kldiv = 0.5*(v_det_term + v_tr_term - n_dim + v_quad_term)

        return mat_kldiv

    def _approx_kldiv_between_diag_gmm_parallel(self, v_vec_alpha_x: torch.Tensor, v_mat_mu_x: torch.Tensor, v_mat_cov_x: torch.Tensor,
                                       v_vec_alpha_y: torch.Tensor, v_mat_mu_y: torch.Tensor, v_mat_cov_y: torch.Tensor):

        v_vec_ln_alpha_x = torch.log(v_vec_alpha_x)
        v_vec_ln_alpha_y = torch.log(v_vec_alpha_y)

        # kldiv_x_x: (n_c_x, n_c_x); kldiv_x_x[i,j] = KL(p_x_i||p_x_j)
        # kldiv_x_y: (n_c_x, n_c_y); kldiv_x_y[i,j] = KL(p_x_i||p_y_j)
        mat_kldiv_x_x = self._kldiv_diag_parallel(v_mat_mu_x, v_mat_cov_x)
        mat_kldiv_x_y = self._kldiv_diag_parallel(v_mat_mu_x, v_mat_cov_x, v_mat_mu_y, v_mat_cov_y)

        # log_sum_pi_exp_c_x: (n_c_x,)
        log_sum_pi_exp_c_x = torch.logsumexp(v_vec_ln_alpha_x.reshape(-1,1) - mat_kldiv_x_x, dim=-1)
        # log_sum_pi_c_x_y: (n_c_x,)
        log_sum_pi_exp_c_x_y = torch.logsumexp(v_vec_ln_alpha_y.reshape(1,-1) - mat_kldiv_x_y, dim=-1)

        kldiv = torch.sum((log_sum_pi_exp_c_x - log_sum_pi_exp_c_x_y)*v_vec_alpha_x)

        return kldiv

    def _marginalize_gaussian_mixture(self, lst_vec_alpha: List[torch.Tensor], lst_mat_mu: List[torch.Tensor], lst_mat_std: List[torch.Tensor],
                                      lst_weight: List[float]):

        # \alpha = \sum_{b}{w_b * \alpha_b}: (\sum{N_component})
        lst_vec_alpha_w = [vec_alpha * w for vec_alpha, w in zip(lst_vec_alpha, lst_weight)]
        vec_alpha = torch.cat(lst_vec_alpha_w)
        # \mu: (\sum{N_component}, N_dim)
        mat_mu = torch.cat(lst_mat_mu, dim=0)
        # \sigma: (\sum{N_component}, N_dim) or (\sum{N_component}, 1)
        mat_std = torch.cat(lst_mat_std, dim=0)

        return vec_alpha, mat_mu, mat_std

    def forward(self, lst_vec_alpha_x: List[torch.Tensor], lst_mat_mu_x: List[torch.Tensor], lst_mat_std_x: List[torch.Tensor],
                vec_alpha_y: torch.Tensor, mat_mu_y: torch.Tensor, mat_std_y: torch.Tensor):
        """
        calculate KL(f_x||f_y); kullback-leibler divergence between two gaussian distributions parametrized by $mu,\Sigma$
        but $\Sigma$ is diagonal matrix.

        :param lst_vec_alpha_x: posterior. list of the packed scale vectors.
        :param lst_mat_mu_x: posterior. list of the sequence of packed mean vectors
        :param lst_mat_std_x: posterior. list of the sequence of packed standard deviation vectors(=sqrt of the diagonal elements of covariance matrix)
        :return: mean of the approximated kl divergence * scale parameter
        """

        lst_seq_len = [len(vec_alpha) for vec_alpha in lst_vec_alpha_x]
        lst_weight = self._generate_weight(lst_seq_len=lst_seq_len)

        if self._marginalize_posterior:
            vec_alpha_x, mat_mu_x, mat_std_x = self._marginalize_gaussian_mixture(lst_vec_alpha_x, lst_mat_mu_x, lst_mat_std_x, lst_weight)
            v_kldiv = self._approx_kldiv_between_diag_gmm_parallel(v_vec_alpha_x=vec_alpha_x, v_mat_mu_x=mat_mu_x, v_mat_cov_x=mat_std_x**2,
                                                          v_vec_alpha_y=vec_alpha_y, v_mat_mu_y=mat_mu_y, v_mat_cov_y=mat_std_y**2)

            return v_kldiv * self._scale * self._multiplier

        else:
            weight_sum = 0
            v_kldiv = torch.zeros(1, dtype=torch.float, requires_grad=True, device=self._device)
            for v_alpha, v_mu, v_std, weight in zip(lst_vec_alpha_x, lst_mat_mu_x, lst_mat_std_x, lst_weight):
                v_kldiv_b = self._approx_kldiv_between_diag_gmm_parallel(v_vec_alpha_x=v_alpha, v_mat_mu_x=v_mu, v_mat_cov_x=v_std**2,
                                                                v_vec_alpha_y=vec_alpha_y, v_mat_mu_y=mat_mu_y, v_mat_cov_y=mat_std_y**2)
                if not torch.isnan(v_kldiv_b):
                    v_kldiv = v_kldiv + v_kldiv_b * weight
                    weight_sum += weight

            if weight_sum > 0:
                v_kldiv = torch.div(v_kldiv, weight_sum) * self._scale * self._multiplier

            return v_kldiv
