#!/usr/bin/env python
# -*- coding:utf-8 -*-
from matplotlib import pyplot as plt

def tile_plot(mat_x):
    fig, ax = plt.subplots()
    ax.imshow(mat_x, interpolation='nearest')
    numrows, numcols = mat_x.shape

    def format_coord(x, y):
        col = int(x + 0.5)
        row = int(y + 0.5)
        if col >= 0 and col < numcols and row >= 0 and row < numrows:
            z = mat_x[row, col]
            return 'x=%1.4f, y=%1.4f, z=%1.4f' % (x, y, z)
        else:
            return 'x=%1.4f, y=%1.4f' % (x, y)

    ax.format_coord = format_coord
    plt.show()