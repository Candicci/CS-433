# -*- coding: utf-8 -*-
"""Exercise 3.

Least Square
"""

import numpy as np
from costs import compute_mse

def least_squares(y, tx):
    """calculate the least squares solution. Returns (w, mse)."""
    a = tx.T @ tx                      # A = XᵀX   (D×D)
    b = tx.T @ y                       # b = Xᵀy   (D,)
    w = np.linalg.solve(a, b)          # résout A w = b, SANS inverser A
    e = y - tx @ w
    mse = e @ e / (2 * len(y))         # même convention 1/(2N) que le lab 2
    return w, mse
