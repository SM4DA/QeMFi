#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  8 12:16:28 2024

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
Script to generate unsorted Global SLATM representations for all 15,000 samples of a given molecule from the CheMFi database.
'''

import numpy as np
import qml
from qml.representations import get_slatm_mbtypes
from tqdm import tqdm
import argparse
np.int = int #needed for qml compatibility
parser = argparse.ArgumentParser()
parser.add_argument("-m", "--molecule", 
                    help="Name of the molecule from the CheMFi dataset. Default is urea.",
                    type=str, default='urea')
parser.add_argument("-d", "--directorypath", 
                    help="Path to the database of molecule including the last '/'. Default is '../dataset/'.",
                    type=str, default='../dataset/')
args = parser.parse_args()


def slatm_glob(molname:str='urea',data_path:str='../dataset/'):
    '''
    Function to generate global SLATM using qml package.

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
    npz = np.load(f'{data_path}CheMFi_{molname}.npz',allow_pickle=True)
    n = npz['R'].shape[0]
    Zs = npz['Z']
    coords = npz['R']
    
    compounds = []
    for i in tqdm(range(n),desc='loading compounds...'):
        comps = qml.Compound(xyz=None)
        comps.coordinates=coords[i]
        comps.nuclear_charges=Zs
        compounds.append(comps)
    
    mbtypes = get_slatm_mbtypes(np.array([mol.nuclear_charges for mol in tqdm(compounds, desc='get mbtype...')]))
    
    for i, mol in tqdm(enumerate(compounds),desc='Generate SLATM...'):
        mol.generate_slatm(mbtypes,local=False)
    
    X_slat_glob = np.asarray([mol.representation for mol in tqdm(compounds,desc='Saving Reps...')])
    
    np.save(f'{molname}_SLATM.npy', X_slat_glob)

slatm_glob(molname=args.molecule,data_path=args.directorypath)

