#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  8 11:45:39 2024

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
Script to generate learning curve data of MFML and o-MFML for the different molecules of MF24 database
'''

import numpy as np
from tqdm import tqdm
from Model_MFML import ModelMFML as MFML
from sklearn.utils import shuffle
import qml.kernels as k
from qml.math import cho_solve
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
parser.add_argument("-l", "--level", 
                    help="Optional index value used only for level in excitation energy properties such as energies or fosc. Default is set to 0, that is , the first component. ",
                    type=int, default=0)
parser.add_argument("-c", "--component", 
                    help="Optional index value used for component of vector properties such as dipole moments. Default is set to 0, that is , the x component.",
                    type=int, default=0)
parser.add_argument("--centeroffset", action=argparse.BooleanOptionalAction,
                    help="Optional bool value to determine whether data should be mean centered for each fidelity. Default is True.",
                    type=bool, default=True)
###MFML related args
parser.add_argument("-n", "--navg", 
                    help="Optional value of number of average runs of the learning curve. Default is set to 1.",
                    type=int, default=1)
parser.add_argument("-w", "--width", 
                    help="Width of kernel being used for KRR. Default is set to 30.",
                    type=float, default=30.0)
parser.add_argument("-rep", "--representation", 
                    help="Molecular descriptor/Representation. Default is set to 'CM'.",
                    type=str, default='CM')
parser.add_argument("-k", "--kernel", 
                    help="Kernel being used for the various KRR models. Default is set to 'matern'.",
                    type=str, default='matern')
parser.add_argument("-r", "--regularizer", 
                    help="Regularization strength to be used in KRR. Default is set to 1-10.",
                    type=float, default=1e-10)
parser.add_argument("-s","--seed", 
                    help="Seed value for training-test split. Default is 42 because that is the answer to everything.",
                    type=int, default=42)

args = parser.parse_args()
########

def data_extraction(molname:str='urea',seed:int=42):
    '''
    Function to extract the data in the multifidelity format to be used in MFML and o-MFML learning curves.
    The first entry of y_trains is the lowest fidelity (in this case STO3G)

    Parameters
    ----------
    molname : str, optional
        Name of molecule of MF24 database. This will be taken from the args parser. The default is 'urea'.
    seed : int, optional
        Seed value for the train-test split. This will be taken from the args parser. The default is 42.

    Returns
    -------
    X_train : np.ndarray
        Training reps.
    X_val : np.ndarray
        Validation reps.
    X_test : np.ndarray
        Test reps.
    train_energies : np.ndarray
        Full fidelity collection of training energies.
    y_val : np.ndarray
        validation energies at target fidelity (in this case, TZVP).
    y_test : np.ndarray
        test energies at target fidelity (in this case TZVP).

    '''
    dataset = np.load(f'{args.directorypath}CheMFi_{molname}.npz',allow_pickle=True)[f'{args.property}']
    #special cases of data. Pick the correct property for KRR.
    if args.property=='TrDP':
        dataset = dataset[:,:,args.level,args.component]
    elif args.property=='EV' or args.property=='fosc':
        dataset = dataset[:,:,args.level]
    elif args.property=='SCF':
        pass
    else:
        dataset=dataset[:,:,args.component]
    
    #load correct rep for molecule and type of rep
    X_full = np.load(f'{molname}_{args.representation}.npy')
    
    #train energies
    train_energies = np.zeros((5),dtype=object)
    for i in range(5):
        if args.centeroffset:
            temp = dataset[:,i]-np.mean(dataset[:,i])
        else:
            temp = dataset[:,i]
        some_y, X = shuffle(temp,X_full,random_state=seed)
        train_energies[i] = some_y[:12288]
    #val and test 
    y_val = some_y[12288:-2000]
    y_test = some_y[-2000:]
    
    #reps
    X_train = X[:12288]
    X_val = X[12288:-2000]
    X_test = X[-2000:]
    
    return X_train, X_val, X_test, train_energies, y_val, y_test


def KRR(X_train:np.ndarray, X_test:np.ndarray, y_train:np.ndarray, y_test:np.ndarray, sigma:float, reg: float):
    '''
    Function to perform single fidelity KRR for a given kernel choice (taken from args parser). 
    It returns mean absolute error (MAE) over a test set.
    For N samples in the test set, MAE are calculated as:
    
    $$MAE = \frac{1}{N}\sum_{i=1}^N \lvert y_i^{ref} - y_i^{predicted} \rvert $$

    Parameters
    ----------
    X_train : np.ndarray
        Training reps.
    X_test : np.ndarray
        reps of test set.
    y_train : np.ndarray
        training energies at single fidelity.
    y_test : np.ndarray
        Reference energies of test set.
    sigma : float
        kernel width. Kernel is chosen from args parser
    reg : float
        Lavrentiev regularizer for KRR.

    Returns
    -------
    mae :np.ndarray
        Mean absolute error 

    '''
    #generate the correct kernel matrix as prescribed by args parser
    if args.kernel=='matern':
        K_train = k.matern_kernel(X_train,X_train,sigma, order=1, metric='l2')
        K_test = k.matern_kernel(X_train,X_test,sigma, order=1, metric='l2')
    elif args.kernel=='laplacian':
        K_train = k.laplacian_kernel(X_train,X_train,sigma)
        K_test = k.laplacian_kernel(X_train,X_test,sigma)
    elif args.kernel=='gaussian':
        K_train = k.gaussian_kernel(X_train,X_train,sigma)
        K_test = k.gaussian_kernel(X_train,X_test,sigma)
    elif args.kernel=='linear':
        K_train = k.linear_kernel(X_train,X_train)
        K_test = k.linear_kernel(X_train,X_test)
    elif args.kernel=='sargan':
        K_train = k.sargan_kernel(X_train,X_train,sigma,gammas=None)
        K_test = k.sargan_kernel(X_train,X_test,sigma,gammas=None)
    
    #regularize 
    K_train[np.diag_indices_from(K_train)] += reg
    #train
    alphas = cho_solve(K_train,y_train)
    #predict
    preds = np.dot(alphas, K_test)
    #MAE calculation
    mae = np.mean(np.abs(preds-y_test))
    
    return mae

def SF_learning_curve(X_train:np.ndarray, X_test:np.ndarray, y_train:np.ndarray, y_test:np.ndarray, 
                      sigma:float=30, reg:float=1e-10, navg:int=10):
    '''
    Function to return the MAE values for different training set sizes, i.e., the learning curve raw data.

    Parameters
    ----------
    X_train : np.ndarray
        Training reps.
    X_test : np.ndarray
        Reps of test set.
    y_train : np.ndarray
        training energies of single fidelity.
    y_test : np.ndarray
        Reference energies of the test set.
    sigma : float, optional
        Kernel width. The default is 30.
    reg : float, optional
        Reghularization for KRR. The default is 1e-10.
    navg : int, optional
        Number of avg runs to perform in the learning curve generation. The default is 10.
    
    Returns
    -------
    full_maes : np.ndarray
        MAEs for training set sizes [2,4,..,512].

    '''
    full_maes = np.zeros((9),dtype=float)
    for n in range(navg):
        maes = []
        X_train,y_train = shuffle(X_train, y_train, random_state=42)
        for i in range(1,10):
            #start_time = time.time()
            temp = KRR(X_train[:2**i],X_test,y_train[:2**i],y_test,sigma=sigma,reg=reg)
            maes.append(temp)
        full_maes += np.asarray(maes)
    
    full_maes = full_maes/navg
    return full_maes

def LC_routine(y_trains:np.ndarray, indexes:np.ndarray, X_train:np.ndarray, X_test:np.ndarray, X_val:np.ndarray, 
               y_test:np.ndarray, y_val:np.ndarray, sigma:float, reg:float, navg:int=10):
    '''
    Function to generate MFML and o-MFML learning curves for 4 baseline fidelities. 

    Parameters
    ----------
    y_trains : np.ndarray
        training energies of 5 fidelities as returned by the data_extraction function.
    indexes : np.ndarray
        Indexes of the features and the properties for multifidelity structure. They indicate which rep point correspodns to which feature in the data structure.
    X_train : np.ndarray
        Training Reps.
    X_test : np.ndarray
        Reps of test set.
    X_val : np.ndarray
        Reps of validation set.
    y_test : np.ndarray
        Reference energies of test set at target fidelity.
    y_val : np.ndarray
        Reference energies of validation set at target fidelity.
    sigma : float
        Kernel width.
    reg : float
        Regularization for KRR.
    navg : int, optional
        Number of runs to avg over for learnign curve generation. The default is 10.

    Returns
    -------
    MAEs_OLS : np.ndarray
        MAEs of OLS optimized o-MFML.
    MAEs_def : np.ndarray
        MAEs of default MFML.

    '''
    nfids = y_trains.shape[0]
    #print(indexes.shape,indexes[0].shape, nfids)

    MAEs_OLS = np.zeros((9),dtype=float) #for OLS MFML
    MAEs_def = np.zeros((9),dtype=float) # for default MFML
    
    for i in tqdm(range(navg),desc='Averaged Learning Curve...', leave=False):
        mae_ntr_OLS = []
        mae_ntr_def = []
        for j in tqdm(range(1,10),leave=False, desc='Loop over training sizes at TZVP'):
            n_trains = np.asarray([2**(j+4),2**(j+3),2**(j+2),2**(j+1),2**j])[5-nfids:]
            ###TRAINING######
            model = MFML(reg=reg, kernel=args.kernel, 
                         order=1, metric='l2', 
                         sigma=sigma, p_bar=False)
            
            model.train(X_train_parent=X_train, 
                        y_trains=y_trains, indexes=indexes, 
                        shuffle=True, n_trains=n_trains, seed=i)
            ######default#########
            predsdef = model.predict(X_test = X_test, y_test = y_test, 
                                     X_val = X_val, y_val = y_val, 
                                     optimiser='default')
            mae_ntr_def.append(model.mae)
            ##########OLS##########
            predsOLS = model.predict(X_test = X_test, y_test = y_test, 
                                     X_val = X_val, y_val = y_val, 
                                     optimiser='OLS', copy_X= True, 
                                     fit_intercept= False)
            mae_ntr_OLS.append(model.mae)
            
            
        #store each avg run MAE  
        mae_ntr_OLS = np.asarray(mae_ntr_OLS)
        mae_ntr_def = np.asarray(mae_ntr_def)
        
        MAEs_OLS += mae_ntr_OLS
        MAEs_def += mae_ntr_def
        
    #return averaged MAE
    MAEs_OLS = MAEs_OLS/navg
    MAEs_def = MAEs_def/navg
    return MAEs_OLS, MAEs_def


def varying_baselines(molname:str, sig:float, reg:float=1-10, navg:int=1):
    '''
    Function to generate MFML learning curves for varying baseline fidelities

    Parameters
    ----------
    molname : str
        Name of moelcule from the MF24 database.
    sig : float
        Kernel width for KRR.
    reg : float, optional
        Regularization for KRR. The default is 1-10.
    navg : int, optional
        Number of runs to average over for leanring curve generation. The default is 1.

    Returns
    -------
    None.
    MAE fiels are saved to working directory.

    '''
    #fix a test set and a validation set
    X_train, X_val, X_test, y_trains, y_val, y_test = data_extraction(molname, seed=args.seed)
    #in y_trains, STO3G comes first, TZVP is the last object array
    
    #run single fidelity KRR for given molecule
    sf_maes = SF_learning_curve(X_train=X_train, X_test=X_test, 
                                y_train=y_trains[-1], y_test=y_test, 
                                sigma=sig, reg=reg,
                                navg=navg) 
    
    np.save(f'SF_{molname}_{args.property}_{args.level}_{args.component}.npy', sf_maes)
    
    for fb in tqdm(range(4),desc='Baseline loop...'):
        indexes = np.load('indexes.npy',allow_pickle=True)
        maeols,maedef = LC_routine(y_trains=y_trains[fb:], indexes=indexes[fb:], 
                                   X_train=X_train, X_test=X_test, 
                                   X_val=X_val, y_test=y_test, y_val=y_val, 
                                   sigma=sig, reg=reg, navg=navg)
        np.save(f'OLS_{molname}_{args.property}_{args.level}_{args.component}_{str(fb)}.npy',maeols)
        np.save(f'def_{molname}_{args.property}_{args.level}_{args.component}_{str(fb)}.npy',maedef)
    
## run the process
varying_baselines(args.molecule, sig=args.width, reg=args.regularizer, navg=args.navg)