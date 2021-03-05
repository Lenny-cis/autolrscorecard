# -*- coding: utf-8 -*-
"""
Created on Sun Jan 24 14:08:39 2021

@author: linjianing
"""


def param_in_validate(param, std_params, errors):
    """参数验证."""
    assert (param in std_params), errors.format(param, '[' + ','.join(std_params) + ']')


def param_contain_validate(param, std_params, errors):
    """参数验证."""
    assert len(set(list(param)).difference(list(std_params))) <= 0,\
        errors.format(param, '[' + ','.join(std_params) + ']')
