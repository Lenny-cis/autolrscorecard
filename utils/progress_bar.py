# -*- coding: utf-8 -*-
"""
Created on Mon Jul 26 23:39:07 2021

@author: Lenny
"""

import sys
from tqdm import tqdm


# PBAR_FORMAT = "Possible: {total}|Elapsed: {elapsed}|Progress: {l_bar}{bar}"
PBAR_FORMAT = "{desc}: {total}|{elapsed}: {percentage:.0f}%|{bar}"


def make_tqdm_iterator(**kwargs):
    """产生tqdm进度条迭代器."""
    options = {
        "file": sys.stdout,
        "leave": True,
        'bar_format': PBAR_FORMAT
    }
    options.update(kwargs)
    iterator = tqdm(**options)
    return iterator
