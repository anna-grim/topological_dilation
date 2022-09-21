# -*- coding: utf-8 -*-
"""
Created on Fri Aug 26 16:00:00 2022

@author: anna.grim
"""

import os
import numpy as np
from copy import deepcopy as cp
from scipy.ndimage.morphology import distance_transform_edt
from tifffile import imwrite


class RegionGraph(object):

    def __init__(self, binary_volume):
        self.input = binary_volume
        self.dist_transform = distance_transform_edt(binary_volume)
        self.global_bdd = self.get_bdd()

        self.dtype = self.input.dtype
        self.nodes, self.super_node_id = self.init_nodes()
        self.super_nodes = self.get_super_nodes()
        self.num_nodes = len(self.nodes)
    
    def get_super_nodes(self):
        return self.nodes[self.super_node_id].mask
    
    def get_bdd(self):
        bdd = np.zeros(self.input.shape)
        bdd[self.dist_transform==1] = 1
        return bdd
        
    def init_nodes(self):
        nodes = dict()
        max_cluster_id = -1
        max_cluster_size = 0
        
        for i in [i for i in np.unique(self.input) if i>0]:
            nodes[i] = Node(self.input, self.global_bdd, self.dtype, i)
            if nodes[i].size>max_cluster_size:
                max_cluster_id = cp(i)
                max_cluster_size = cp(nodes[i].size)
        return nodes, max_cluster_id

    def get_best_merge(self):
        min_nb_dist = np.inf
        min_nb_id = -1
        best_pair = -1
        for i in [j for j in self.nodes if j!=self.super_node_id]:
            dist, pair = self.compute_dist(i, self.super_node_id) 
            if dist<=3:
                return i, pair
            elif dist<min_nb_dist:
                min_nb_dist = cp(dist)
                min_nb_id = cp(i)
                best_pair = cp(pair)

        return min_nb_id, best_pair
    
    def compute_dist(self, i, j):
        best_pair = []
        min_dist = np.inf
        bdd_i = self.nodes[i].bdd
        bdd_j = self.nodes[j].bdd
        for k in range(bdd_i.shape[1]):
            for l in range(bdd_j.shape[1]):
                dist = self.l1_metric(bdd_i[:,k], bdd_j[:,l])
                if dist<min_dist:
                    min_dist = cp(dist)
                    best_pair = [bdd_i[:,k], bdd_j[:,l]]
        return min_dist, best_pair

    def l1_metric(self, x, y):
        return np.sum(abs(x-y))
    
    def upd_super_node(self, key):
        super_id = cp(self.super_node_id)
        self.super_nodes = np.logical_or(self.super_nodes, self.nodes[key].mask)
        self.nodes[super_id].mask = self.super_nodes 
        self.nodes[super_id].bdd = self.nodes[super_id].get_bdd_voxels(self.dist_transform)
        del self.nodes[key]
        self.num_nodes -= 1


class Node:

    def __init__(self, ccp, dist_transform, dtype, key):
        self.id = key
        self.dtype = dtype
        self.mask = self.get_mask(ccp)
        self.size = np.sum(self.mask)
        self.bdd = self.get_bdd_voxels(dist_transform)

    def get_mask(self, ccp):
        mask = np.zeros((ccp.shape), dtype=self.dtype)
        mask[ccp==self.id] = 1
        return mask

    def get_bdd_voxels(self, dist_transform):
        bdd = np.zeros((self.mask.shape))
        x = self.mask * dist_transform
        bdd[x==1] = 1
        return np.array([arr for arr in np.where(bdd>0)])