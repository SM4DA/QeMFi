#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  8 10:02:34 2024

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
Script to generate preliminary analysis plots for the various data of CheMFi database.
'''

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("-m", "--molecule", 
                    help="Name of the molecule from the CheMFi dataset. Default is urea.",
                    type=str, default='urea')
parser.add_argument("-d", "--directorypath", 
                    help="Path to the database of molecule including the last '/'. Default is '../dataset/'.",
                    type=str, default='../dataset/')
parser.add_argument("-p", "--property", 
                    help="Property of interest for which prelim analysis is to be performed. use data.files to check for valid keys in the CheMFi dataset. Default is SCF.", 
                    type=str, default='SCF')
parser.add_argument("-u", "--units", 
                    help="Unity of the property of interest for which prelim analysis is to be performed. use data.files to check for valid keys in the CheMFi dataset. Default is units.", 
                    type=str, default='units')
parser.add_argument("-l", "--level", 
                    help="Optional index value used only for level in excitation energy properties such as energies or fosc. Default is set to 0, that is , the first level. ",
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
args = parser.parse_args()


def prelim_analysis(data:np.ndarray,center:bool=True,units:str=args.units):
    '''
    Function to perform preliminary analysis and plot the results

    Parameters
    ----------
    data : np.ndarray
        data array of size (n_samples,5).
    center : bool, optional
        Whether to center the data for each fidelity by the mean or not. The default is True.
    units : str, optional
        Units of hte property, used in the plotting only. The default is 'units'.

    Returns
    -------
    fig : plt.figure  
        matplotlib figure object.

    '''
    std = []
    diffs = []
    
    fids = ['STO3G','3-21G','6-31G','SVP','TZVP']
    
    fig,ax = plt.subplots(1,3,figsize=(12,4))
    fig.tight_layout(pad=3)
    
    for i in range(5):
        if center:
            data[:,i] = data[:,i]-np.mean(data[:,i])
        sns.kdeplot(ax=ax[0], data=data[:,i], 
                    common_norm=True, common_grid=True,
                    bw_method='scott', levels=10, thresh=0.05, 
                    gridsize=200, cut=3,
                    fill=False,bw_adjust=1,
                    cumulative=False,
                    label=fids[i])
    ax[0].legend()
    #diff and std
    for i in range(4):
        diffs.append(
            np.mean(np.abs(data[:,i]-data[:,-1]))
        )
        std.append(
            np.std(np.abs(data[:,i]-data[:,-1]))
        )
        
        ax[2].scatter(data[:,i],data[:,-1],s=2,label=fids[i])
    ax[2].plot([np.min(data[:,-1]),np.max(data[:,-1])],[np.min(data[:,-1]),np.max(data[:,-1])],linestyle='--',color='k')
    ax[0].ticklabel_format(axis='both',scilimits=[-1,1])
    ax[1].ticklabel_format(axis='y',scilimits=[0,0])
    ax[2].ticklabel_format(axis='both',scilimits=[-1,1])
        
    ax[1].errorbar(fids[:-1],diffs,std,color='k',ecolor='red',capsize=3,elinewidth=1,barsabove=True)
    ax[1].set_ylabel('$\Delta y_f^F$'+f'[{units}]')
    
    ax[2].set_xlabel(r'$y^{f}$'+f' [{units}]')
    ax[2].set_ylabel(r'$y^{TZVP}$'+ f' [{units}]')
    ax[0].set_xlabel(r'$y^f$'+f' [{units}]')
    
    ax[0].set_title('Distribution',weight='bold')
    ax[1].set_title('Fidelity Difference',weight='bold')
    ax[2].set_title('Fidelity Scatter',weight='bold')
    ax[2].legend(markerscale=3)
    
    return fig

#load database
data = np.load(f'{args.directorypath}CheMFi_{args.molecule}.npz')[f'{args.property}']

#make prelim checks for size of data arrays
#make corresponding data array and perform analysis
if args.property=='R' or args.property=='Z' or args.property=='CONF' or args.property=='ID':
    print('Cannot perform prelim analysis for this property since it does not have a multifidelity structure.')
elif args.property=='SCF':
    figout=prelim_analysis(data,center=args.centeroffset)
elif args.property=='TrDP':
    data = data[:,:,args.level,args.component]
    figout=prelim_analysis(data,center=args.centeroffset)
elif args.property=='EV' or args.property=='fosc':
    data = data[:,:,args.level]
    figout=prelim_analysis(data,center=args.centeroffset)
else:
    data=data[:,:,args.component]
    figout=prelim_analysis(data,center=args.centeroffset)

if args.saveplot:
    figout.savefig(f'{args.molecule}_{args.property}_{args.level}_{args.component}_prelim_analysis.pdf',bbox_inches='tight',format='pdf')
