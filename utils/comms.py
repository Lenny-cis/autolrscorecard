# -*- coding: utf-8 -*-
"""
Created on Mon Jan 25 16:35:58 2021

@author: linjianing
"""

import ray


def update_dict_value(orient_dict, new_dict, func):
    """根据func更新嵌套字典最内层."""
    for key, val in orient_dict.items():
        if isinstance(val, dict):
            yield from [(key, dict(update_dict_value(val, new_dict, func)))]
        elif key in new_dict.keys():
            yield (key, func(val, new_dict[key]))
        else:
            yield (key, val)
