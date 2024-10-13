#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  8 10:13:27 2024

@author: vvinod
MIT License

Copyright (c) [2024] [Vivin Vinod]

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""
'''
Script to generate unsorted CM representations for all 15,000 samples of a given molecule from the QeMFi database.
'''

import numpy as np
import qml
from tqdm import tqdm
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("-m", "--molecule", 
                    help="Name of the molecule from the QeMFi dataset. Default is urea.",
                    type=str, default='urea')
parser.add_argument("-d", "--directorypath", 
                    help="Path to the database of molecule including the last '/'. Default is '../dataset/'.",
                    type=str, default='../dataset/')
parser.add_argument("-s", "--sorting", 
                    help="Sorting method to be used while generating CM. Default is 'unsorted'. Alternative can be 'row-norm'. See qmlcode.org for more details.",
                    type=str, default='unsorted')
args = parser.parse_args()


def generate_CM(molname:str='urea',data_path:str='../dataset/'):
    '''
    Function to generate unsorted CM using qml package.

    Parameters
    ----------
    molname : str, optional
        Name of the molecule. The default is 'urea'.
    data_path : str, optional
        Full path to the molecule database. The default is '../dataset/'.

    Returns
    -------
    None.

    '''
    npz = np.load(f'{data_path}QeMFi_{molname}.npz',allow_pickle=True)
    n = npz['R'].shape[0]
    reps = []
    Zs = npz['Z']
    coords = npz['R']
    for i in tqdm(range(n),desc='Generating CM', leave=False):
        temp = qml.Compound(xyz=None)
        temp.coordinates=coords[i]
        temp.nuclear_charges = Zs
        temp.generate_coulomb_matrix(size=Zs.shape[0],sorting=args.sorting)
        reps.append(temp.representation)
    reps = np.asarray(reps)
    np.save(f'{molname}_{args.sorting}_CM.npy',reps)


generate_CM(molname=args.molecule,data_path=args.directorypath)

