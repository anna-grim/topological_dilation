# -*- coding: utf-8 -*-
"""
Created on Fri Aug 26 12:00:00 2022

This code dilates neurons from image volumes in manner that preserves the topology.
The 26-connected components are detected for each neuron, then checks whether each
component is 6-connected. When the connectivity differs, a minimal number of voxels
are added until the given 26-connected component is 6-connected.

@author: anna.grim
"""

import os
import numpy as np
from copy import deepcopy as cp
from cc3d import connected_components
from digital_topology import check_simple
from dijkstra3d import dijkstra as get_ShortestPath
from region_graph import RegionGraph
from skimage.io import imread
from tifffile import imwrite


def get_mask(arr, key):
    """
    Computes binary mask in which non-zero entries correspond to "arr" entries with value "key"

        :param arr: numpy array
        :param key: values from arr which are non-zero in mask
    """
    mask = np.zeros(arr.shape, dtype=arr.dtype)
    mask[arr==key] = 1
    return mask

def get_window(arr, bbox=[]):
    """
    Extracts subarr from "arr" using "bbox" to locate the region of interest

        :param arr: numpy 3d array
        :param bbox: bounding box
    """
    if len(bbox)<1:
        bbox1, bbox2 = get_bounding_box(arr)
    else:
        bbox1 = bbox[0]
        bbox2 = bbox[1]
    subarr = arr[bbox1[0]:bbox2[0], bbox1[1]:bbox2[1], bbox1[2]:bbox2[2]]
    return subarr, [bbox1, bbox2]

def get_bounding_box(arr):
    """
    Extracts bounding box coordinates of non-zero region from a numpy 3d arr

        :param arr: numpy 3d array
    """
    coords1 = []
    coords2 = []
    for i, idxs in enumerate(np.where(arr>0)):
        coords1.append(max(0, np.min(idxs)-2))
        coords2.append(min(arr.shape[i], np.max(idxs)+2))
    return coords1, coords2

def dilate_component(volume, ccp):
    """
    Given a 26-ccp, the component is dilated so that it's 6-ccp. This is 
    done by creating a region graph in which each node corresponds to a 6-ccp,
    then nearest nodes are connecteds by a shortest path.
    
        :param volume: original input volume
        :param ccp: mask of ccp in which each ccp has a unique id
    """
    region_graph = RegionGraph(ccp)
    super_node = region_graph.super_node_id
    while region_graph.num_nodes>1:
        i, voxels = region_graph.get_best_merge()
        ccp = connect_nodes(volume, ccp, voxels, super_node)
        ccp[ccp==i] = super_node     
        region_graph.upd_super_node(i)
    ccp[ccp>0] = 1
    return ccp
        
def connect_nodes(volume, region, voxels, label):
    """
    Given two voxels, this routine finds the shortest path between them such that 
    this path is contained in "region" and each voxel on the path is simple
    
        :param volume: original input volume
        :param region: subarr that represents the region where the path is contained
        :param voxels: list of voxels to connect
        :param label: label given to voxels on the path
    """
    field = np.ones(region.shape)
    while True:
        path = get_ShortestPath(field, voxels[0], voxels[1], connectivity=6)
        if check_path(volume, path):
            for i in path:
                region[tuple(i)] = label
            break
        elif len(path)>4:
            break
        else:
            field[tuple(i)] = 0
    return region

def check_path(volume, path):
    """
    Determines whether each voxel on "path" is simple
    
        :param volume: orginal input volume
        :param path: list of voxels in path
    """
    for i in path:
        nbhd = get_nbhd(volume, tuple(i))
        if not check_simple(nbhd, connectivity=26):
            return False, i
    return True, None

def get_nbhd(arr, idx):
    """
    Gets the 3x3x3 nbhd about "idx" in "arr" 
    
        :param arr: numpy 3d array
        :param idx: index in "arr"
    """
    x,y,z = arr.shape
    nbhd = arr[max(idx[0]-1,0):min(idx[0]+2,x),
               max(idx[1]-1,0):min(idx[1]+2,y),
               max(idx[2]-1,0):min(idx[2]+2,z)]
    nbhd[1,1,1] = 1

    if nbhd.shape!=(3,3,3):
        dims = 3-np.array(nbhd.shape)
        nbhd = np.pad(nbhd, pad_width=((0,dims[0]),(0,dims[1]), (0,dims[2])), mode='reflect')
    return nbhd

def embed_neuron(volume, dilated_neuron, bbox, label):
    """
    Embeds dilated neuron into original input volume
    
        :param volume: original input volume
        :param dilated_neuron: 
        :param bbox: orginal bounding box of "dilated_neuron"
        :param label: label corresponding to "dilated_neuron"
    """
    mask = np.zeros(volume.shape, dtype=bool)
    bbox1 = bbox[0]
    bbox2 = bbox[1]
    mask[bbox1[0]:bbox2[0], bbox1[1]:bbox2[1], bbox1[2]:bbox2[2]] = dilated_neuron
    volume[mask] = label
    return volume

def main(root_dir, input_dir):

    # Initializations
    output_dir = os.path.join(root_dir, 'dilated_inputs')
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    # Main
    fn_list = os.listdir(input_dir)
    fn_list.sort()
    for fn in [fn for fn in fn_list if 'tif' in fn]:
        print(fn)
        labels = imread(os.path.join(input_dir, fn))
        dilated_labels = cp(labels)

        print('   Number of neurons:', len(np.unique(labels))-1)
        for i in [i for i in np.unique(labels) if i!=0]:
            neuron = get_mask(labels, i)
            extracted_neuron, bbox = get_window(neuron)

            ccp26 = connected_components(extracted_neuron, connectivity=26)
            ccp26_ids = [i for i in np.unique(ccp26) if i>0]
            print('      Neuron {} has {} 26-CCPs'.format(i, len(ccp26_ids)))
            for j in ccp26_ids:

                # Fuse disconnected components (if applicable)
                neuron_j = get_mask(ccp26, j)
                ccp6, num_ccp6 = connected_components(neuron_j, connectivity=6, return_N=True)
                if num_ccp6>1:
                    print('         Dilating component', j)
                    volume_window, _ = get_window(dilated_labels, bbox)   
                    dilated_neuron_j = dilate_component(volume_window, ccp6)
                    dilated_labels = embed_neuron(dilated_labels, dilated_neuron_j, bbox, i)

        imwrite(os.path.join(output_dir, fn), dilated_labels)
        print('')


if __name__=="__main__":
    
    root_dir = './'
    input_dir = os.path.join(root_dir, 'inputs')
    main(root_dir, input_dir)

