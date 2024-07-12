#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 12 16:17:23 2024

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
Script to generate learning curve plots.
'''
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import pylab as pl
import argparse
parser = argparse.ArgumentParser()
##args to load data and structure it
parser.add_argument("-m", "--molecule", 
                    help="Name of the molecule from the MF24 dataset. Default is urea.",
                    type=str, default='urea')
parser.add_argument("-d", "--directorypath", 
                    help="Path to the database of molecule including the last '/'. Default is '../dataset/'.",
                    type=str, default='../dataset/')
parser.add_argument("-p", "--property", 
                    help="Property of interest for which prelim analysis is to be performed. use data.files to check for valid keys in the MF24 dataset. Default is SCF.", 
                    type=str, default='SCF')
parser.add_argument("-u", "--units", 
                    help="Unit of the property. This is used in the ylabel of the plot. Default is hE.", 
                    type=str, default='hE')
parser.add_argument("-l", "--level", 
                    help="Optional index value used only for level in excitation energy properties such as energies or fosc. Default is set to 0, that is , the first component. ",
                    type=int, default=0)
parser.add_argument("-c", "--component", 
                    help="Optional index value used for component of vector properties such as dipole moments. Default is set to 0, that is , the x component.",
                    type=int, default=0)
parser.add_argument("--centeroffset", action=argparse.BooleanOptionalAction, 
                    help="Optional bool value to determine whether data should be mean centered for each fidelity. Default is True.",
                    type=bool, default=True)
parser.add_argument("--saveplot", action=argparse.BooleanOptionalAction,
                    help="Optional bool value to save preliminary analysis plot in the current directory as a png file. Default is True.",
                    type=bool, default=True)
parser.add_argument("-rep", "--representation", 
                    help="Molecular descriptor/Representation. Default is set to 'CM'.",
                    type=str, default='CM')

args = parser.parse_args()

def plot_function(sf:np.ndarray, mfml:np.ndarray, omfml:np.ndarray, times:np.ndarray, units:str='units'):
    '''
    Function to plot learning curves of single fidelity KRR, MFML, and o-MFML.

    Parameters
    ----------
    sf : np.ndarray
        Single fidelity MAEs.
    mfml : np.ndarray
        default MFML MAEs.
    omfml : np.ndarray
        MAEs from OLS optimized o-MFML.
    times : np.ndarray
        Compute times for the QC calculations. This will be taken from the CheMFi dataset.
    units : str, optional
        Units of the property. This will be read from the args parser. The default is 'units'.
    

    Returns
    -------
    fig : matplotlib.figure
        Figure object with plots.

    '''
    markers = ['^','*','P','d']
    colors = pl.cm.cividis(np.linspace(0,1,5))
    
    n = 2**np.arange(1,10)
    
    
    
    ####compute time to generate training data
    MFML_times = np.zeros((5),dtype=object)
    temp_t = np.zeros((9),dtype=float)
    
    for f in range(5):
        for i in range(1,10):
            temp_t[i-1] += times[4-f]*(2**i)*(2**f)
        MFML_times[f] = np.copy(temp_t)/60.0    
    ########
    
    y_minor = mticker.LogLocator(base = 10, subs = np.arange(1.0,20,3), numticks = 20)
    y_major = mticker.LogLocator(base = 10, subs = np.arange(1.0,10,3), numticks = 10)
    
    fig,ax= plt.subplots(1,2,figsize=(8,4),sharey=True)
    ax[0].set_title('MFML')
    ax[1].set_title('o-MFML')
    ax[0].set_ylabel(f'MAE [m{units}]')
    
    #sf plots in both subplots
    ax[0].loglog(MFML_times[0],1e3*sf,marker='o',color=colors[0],label='KRR (TZVP)',linestyle=':')
    ax[1].loglog(MFML_times[0],1e3*sf,marker='o',color=colors[0],label='KRR (TZVP)',linestyle=':')
    
    #multyifidelity plots
    for i in range(4):
        ax[0].loglog(MFML_times[i+1], 1e3*mfml[i], color=colors[i+1],marker=markers[i], linestyle='--')
        ax[1].loglog(MFML_times[i+1], 1e3*omfml[i], color=colors[i+1],marker=markers[i], linestyle='-')
    
    #cosmetics of the plots
    for i in range(2):
        ax[i].yaxis.set_minor_formatter(mticker.ScalarFormatter())
        ax[i].yaxis.set_major_formatter(mticker.ScalarFormatter())
        ax[i].yaxis.set_minor_locator(y_minor)
        ax[i].yaxis.set_major_locator(y_major)
        ax[i].grid(True, which="major", ls="-",color='dimgray')
        ax[i].grid(True, axis='y', which="minor", ls="--",color='lightgray')
        ax[i].set_xlabel('$T_{\mathrm{train}}^{\mathrm{MFML}} [min]$')
    fig.legend(['KRR-reference','SVP','6-31G','3-21G','STO-3G'],ncol=5,
               bbox_to_anchor=(0.85,-0.02),title='$f_b$')
    
    return fig

def main():
    sf = np.load(f'SF_{args.molecule}_{args.representation}_{args.property}_{args.level}_{args.component}.npy')
    mfml = np.zeros((4),dtype=object)
    omfml = np.zeros((4),dtype=object)
    
    calc_times = np.load(f'{args.directorypath}CheMFi_{args.molecule}.npz',allow_pickle=True)['t']
    
    for i in range(4):
        omfml[i] = np.load(f'OLS_{args.molecule}_{args.representation}_{args.property}_{args.level}_{args.component}_{3-i}.npy')
        mfml[i] = np.load(f'def_{args.molecule}_{args.representation}_{args.property}_{args.level}_{args.component}_{3-i}.npy')
    figure = plot_function(sf=sf, mfml=mfml, omfml=omfml, units=args.units, times=calc_times)
    
    if args.saveplot:
        figure.savefig(f'{args.molecule}_{args.representation}_{args.property}_{args.level}_{args.component}_time_learning_curves.pdf', bbox_inches='tight',format='pdf')

main()
