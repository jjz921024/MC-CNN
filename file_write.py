#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  6 14:40:56 2018

@author: jun
"""
def write_result(trust, pred):
    with open('test_result', 'w') as f:
        for i in range(len(pred)):
            f.write(trust[i]+'----'+ pred[i] + '\n')
        

