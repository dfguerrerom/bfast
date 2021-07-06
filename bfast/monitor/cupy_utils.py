# copy the utils functionalities into cupy objects
from datetime import datetime

import cupy as cp
import numpy as np
import pandas

from . import utils


__critvals = cp.array(utils.__critvals)
__critval_h = cp.array(utils.__critval_h)
__critval_period = cp.array(utils.__critval_period)
__critval_level = cp.array(utils.__critval_level)
__critval_mr = utils.__critval_mr

check = utils.check

def get_critval(h, period, level, mr):
    
    # Sanity check
    check(h, period, level, mr)

    index = cp.zeros(4, dtype=cp.int)

    # Get index into table from arguments
    index[0] = next(i for i, v in enumerate(__critval_mr) if v == mr)
    index[1] = cp.where(level == __critval_level)[0][0]
    index[2] = (cp.abs(__critval_period - period)).argmin()
    index[3] = cp.where(h == __critval_h)[0][0]
    
    # For legacy reasons, the critvals are scaled by sqrt(2)
    return __critvals[tuple(index)] * cp.sqrt(2)

_find_index_date = utils._find_index_date

def compute_lam(N, hfrac, level, period):
    
    check(hfrac, period, 1 - level, "max")

    return get_critval(hfrac, period, 1 - level, "max")

compute_end_history = utils.compute_end_history

def map_indices(dates):
    
    return cp.array(utils.map_indices(dates))
