# -*- coding: utf-8 -*-

"""
Created on Thu Sep  1 13:31:31 2022

@author: anna.grim
"""

import numpy as np
from cc3d import connected_components


def check_simple(nbhd, connectivity=26):
    assert connectivity==6 or connectivity==26
    if connectivity==6:
        simple6 = topo_number(nbhd, 6) == 1
        simple26 = topo_number(1-nbhd, 26) == 1
    else:
        simple6 = topo_number(1-nbhd, 6) == 1
        simple26 = topo_number(nbhd, 26) == 1
    return True if (simple6 and simple26) else False
    

def topo_number(nbhd, connectivity):
    assert connectivity==6 or connectivity==26
    if connectivity==6:
        nbhd6 = get_nbhd(6)
        nbhd18 = get_nbhd(18)
        ccp = nbhd6*connected_components(nbhd18*nbhd, connectivity=6)
    else:
        nbhd[1,1,1] = 0
        ccp = connected_components(nbhd, connectivity=26)
    return len([i for i in np.unique(ccp) if i!=0])

def get_nbhd(n):
    assert n==6 or n==18
    if n==6:
        nbhd = np.zeros((3,3,3))
        nbhd[:,1,1] = 1
        nbhd[1,:,1] = 1
        nbhd[1,1,:] = 1
    elif n==18:
        nbhd = np.zeros((3,3,3))
        nbhd[:,:,1] = 1
        nbhd[1,:,:] = 1
        nbhd[:,1,:] = 1
        nbhd[1,1,1] = 0
    return nbhd


