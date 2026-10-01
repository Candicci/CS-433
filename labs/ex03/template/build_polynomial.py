# -*- coding: utf-8 -*-
"""implement a polynomial basis function."""

import numpy as np


def build_poly(x, degree):
    """polynomial basis functions for input data x, for j=0 up to j=degree."""
    """Φ : colonne j = x^j, pour j = 0..degree → shape (N, degree+1)"""
    return x[:, None] ** np.arange(degree + 1)

def compute_mse(y, tx, w):
    e = y - tx @ w
    return e @ e / (2 * len(y))          # convention du cours : 1/(2N)
