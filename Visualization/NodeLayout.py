# -*- coding: utf-8 -*-
"""
Created on Thu Apr 28 09:33:19 2022

@author: remit
"""
import numpy as np


def normalisation(data_noeud):
    m=0
    for elem in data_noeud:
        norm=np.sqrt(elem["positionX"]**2+elem["positionY"]**2)
        if norm>m:
            m=norm
            
    for elem in data_noeud:
        elem["positionX"]= elem["positionX"]/m
        elem["positionY"]= elem["positionY"]/m
    return data_noeud