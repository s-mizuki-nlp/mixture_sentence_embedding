#!/usr/bin/env python
# -*- coding:utf-8 -*-
from matplotlib import pyplot as plt
from matplotlib import ticker
from matplotlib import colors
from matplotlib import cm
import numpy as np

_DEFAULT_FIG_SIZE = (8,8)

def tile_plot(mat_dist, fig_and_ax=None, lst_ticker_x=None, lst_ticker_y=None, figsize=None, cmap="Reds", colorbar=True, **kwargs):

    @ticker.FuncFormatter
    def major_formatter(x, pos):
        return f"{x+0.5:2.0f}"

    if fig_and_ax is None:
        fig, ax = plt.subplots(figsize=_DEFAULT_FIG_SIZE if figsize is None else figsize)
    else:
        fig, ax = fig_and_ax[0], fig_and_ax[1]

    n_x, n_y = mat_dist.shape
    mesh_x, mesh_y = np.meshgrid(np.arange(n_x+1), np.arange(n_y+1))

    ax.pcolormesh(mesh_x, mesh_y, mat_dist.T, cmap=cmap, **kwargs)
    # adjust axes labels
    ## X=axis
    ax.xaxis.set_major_locator(ticker.FixedLocator(np.arange(0.5, n_x, 1.0)))
    if lst_ticker_x is None:
        ax.xaxis.set_major_formatter(major_formatter)
    else:
        ax.xaxis.set_major_formatter(ticker.FixedFormatter(lst_ticker_x))
        ax.tick_params(axis="x", rotation=90)
    ## Y-axis
    ax.yaxis.set_major_locator(ticker.FixedLocator(np.arange(0.5, n_y, 1.0)))
    if lst_ticker_y is None:
        ax.yaxis.set_major_formatter(major_formatter)
    else:
        ax.yaxis.set_major_formatter(ticker.FixedFormatter(lst_ticker_y))
    ## colorbar
    if colorbar:
        norm = colors.Normalize(vmin=mat_dist.min(), vmax=mat_dist.max())
        mappable = cm.ScalarMappable(cmap=cmap, norm=norm)
        mappable._A = []
        fig.colorbar(mappable)

    return fig, ax