# -*- coding: utf-8 -*-
"""a function used to compute the loss."""

import numpy as np


def compute_loss(y, tx, w):
    """Calculate the loss using either MSE or MAE.

    Args:
        y: numpy array of shape=(N, )
        tx: numpy array of shape=(N,2)
        w: numpy array of shape=(2,). The vector of model parameters.

    Returns:
        the value of the loss (a scalar), corresponding to the input parameters w.
    """
    e = y - tx.dot(w)
    return e.dot(e) / (2 * len(y))
