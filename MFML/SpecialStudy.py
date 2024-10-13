import numpy as np
from tqdm.notebook import tqdm
from Model_MFML import ModelMFML as MFML
from sklearn.utils import shuffle
import qml.kernels as k
from qml.math import cho_solve

def prep_data():
    molnames = ['urea','acrolein','alanine','sma','nitrophenol','urocanic','dmabn','thymine','o-hbdi']
    idx = np.arange(0,15000)
    idx = shuffle(idx,random_state=42)
    idx=idx[:1500]
    
    X=np.zeros((13500,6438),dtype=float) #largest SLATM is for o-hbdi with 6438 features. Rest will be padded to this size.
    y_all = np.zeros((13500,5),dtype=float)
    
    start=0
    end=1500
    idx_names = np.zeros((13500),dtype=float)
    
    for i,m in tqdm(enumerate(molnames),desc='building composite dataset of uncentered SCF'):
        names = np.full(1500,i)
        idx_names[start:end] = np.copy(names)
        temp_X = np.load(f'Reps/{m}_SLATM.npy')
        X[start:end,:temp_X.shape[-1]] = temp_X[idx,:]
        temp_data=np.load(f'../dataset/QeMFi_{m}.npz')['SCF']
        y_all[start:end,:] = temp_data[idx,:]
        
        #increment for next molecule
        start+= 1500
        end += 1500
    
    y_new = np.zeros((5),dtype=object)
    
    X,idx_names = shuffle(X,idx_names,random_state=42)
    y_train = np.zeros((5),dtype=object)
    for i in range(5):
        y_all[:,i] = shuffle(y_all[:,i],random_state=42)
        y_new[i] = y_all[:,i]
        y_train[i] = y_new[i][:11000]
    
    X_train = X[:11000,:]
    X_val = X[11000:11500,:]
    y_val = y_new[-1][11000:11500]
    X_test = X[11500:,:]
    y_test = y_new[-1][11500:]
    
    return X_train, X_val, X_test, y_train, y_val, y_test, idx_names
    
def KRR(X_train:np.ndarray, X_test:np.ndarray, y_train:np.ndarray, y_test:np.ndarray, k_type:str='laplacian', sigma:float=30.0, reg:float=1e-10):
    if k_type=='matern':
        K_train = k.matern_kernel(X_train,X_train,sigma, order=1, metric='l2')
        K_test = k.matern_kernel(X_train,X_test,sigma, order=1, metric='l2')
    elif k_type=='laplacian':
        K_train = k.laplacian_kernel(X_train,X_train,sigma)
        K_test = k.laplacian_kernel(X_train,X_test,sigma)
    elif k_type=='gaussian':
        K_train = k.gaussian_kernel(X_train,X_train,sigma)
        K_test = k.gaussian_kernel(X_train,X_test,sigma)
    elif k_type=='linear':
        K_train = k.linear_kernel(X_train,X_train)
        K_test = k.linear_kernel(X_train,X_test)
    elif k_type=='sargan':
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
    
    return mae, preds

def SF_learning_curve(X_train:np.ndarray, X_test:np.ndarray, y_train:np.ndarray, y_test:np.ndarray, 
                      k_type:str='laplacian',sigma:float=30, reg:float=1e-10, navg:int=10):
    full_maes = np.zeros((9),dtype=float)
    for n in tqdm(range(navg),desc='average LC generation'):
        maes = []
        X_train,y_train = shuffle(X_train, y_train, random_state=42)
        for i in range(1,10):
            temp, preds = KRR(X_train[:2**i],X_test,y_train[:2**i],y_test,sigma=sigma,reg=reg,k_type=k_type)
            maes.append(temp)
        full_maes += np.asarray(maes)
    
    full_maes = full_maes/navg
    return full_maes, preds

def LC_routine(y_trains:np.ndarray, indexes:np.ndarray, X_train:np.ndarray, X_test:np.ndarray, X_val:np.ndarray, 
               y_test:np.ndarray, y_val:np.ndarray, k_type:str='laplacian',sigma:float=200.0, reg:float=1e-10, navg:int=10):
    nfids = y_trains.shape[0]
    
    MAEs_OLS = np.zeros((9),dtype=float) #for OLS MFML
    MAEs_def = np.zeros((9),dtype=float) # for default MFML
    
    for i in tqdm(range(navg),desc='avg run',leave=False):
        mae_ntr_OLS = []
        mae_ntr_def = []
        for j in range(1,10):
            n_trains = np.asarray([2**(j+4),2**(j+3),2**(j+2),2**(j+1),2**j])[5-nfids:]
            ###TRAINING######
            model = MFML(reg=reg, kernel=k_type, 
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


def varying_baselines(X_train, X_val, X_test, y_train, y_val, y_test, ker:str='laplacian',sig:float=200.0, reg:float=1-10, navg:int=1):
    
    #run single fidelity KRR for given molecule
    sf_maes,sf_preds = SF_learning_curve(X_train=X_train, X_test=X_test, 
                                  y_train=y_train[0], y_test=y_test, k_type=ker,
                                  sigma=200.0, reg=1e-10,
                                  navg=navg) 
    
    np.save('outputs/specialcase/sf_mae.npy',sf_maes)
    np.save('outputs/specialcase/sf_preds.npy',sf_preds)
    
    maeols = np.zeros((4),dtype=object)
    maedef = np.zeros((4),dtype=object)
    
    indexes = np.zeros((5),dtype=object)
    for i in range(5):
        indexes[i] = np.vstack([np.arange(0,11000),np.arange(0,11000)]).T
    
    for fb in tqdm(range(4),desc='Baseline loop...'):
        maeols[fb],maedef[fb] = LC_routine(y_trains=y_train[fb:], indexes=indexes[fb:], 
                                           X_train=X_train, X_test=X_test, 
                                           X_val=X_val, y_test=y_test, y_val=y_val, k_type=ker,
                                           sigma=200.0, reg=1e-10, navg=navg)
    
    np.save('outputs/specialcase/def_mae.npy',maedef)
    np.save('outputs/specialcase/ols_mae.npy',maeols)
    
def main():
    #TZVP first in y_trains
    X_train, X_val, X_test, y_train, y_val, y_test, mol_IDs = prep_data()
    varying_baselines(X_train, X_val, X_test, y_train, y_val, y_test, sig=200.0, reg=1e-10, navg=10)

    
def predict_main():
    X_train, X_val, X_test, y_train, y_val, y_test, mol_IDs = prep_data()
    indexes = np.zeros((5),dtype=object)
    for i in range(5):
        indexes[i] = np.vstack([np.arange(0,11000),np.arange(0,11000)]).T
    
    defpreds = np.zeros((4),dtype=object)
    olspreds = np.zeros((4),dtype=object)
    
    for fb in range(4):
        n_trains = np.asarray([2**13,2**12,2**11,2**10,2**9])[fb:]
        ###TRAINING######
        model = MFML(reg=1e-10, kernel='laplacian', 
                     order=1, metric='l2', 
                     sigma=200.0, p_bar=False)

        model.train(X_train_parent=X_train, 
                    y_trains=y_train[fb:], indexes=indexes[fb:], 
                    shuffle=True, n_trains=n_trains, seed=42)
        #default MFML
        tempdef = model.predict(X_test = X_test, y_test = y_test, 
                                X_val = X_val, y_val = y_val, 
                                optimiser='default')
        defpreds[fb] = np.copy(tempdef)
        #OLS o-MFML
        tempols = model.predict(X_test = X_test, y_test = y_test, 
                                 X_val = X_val, y_val = y_val, 
                                 optimiser='OLS', copy_X= True, 
                                 fit_intercept= False)
        olspreds[fb]=np.copy(tempols)
    
    np.save('outputs/specialcase/def_preds.npy',defpreds)
    np.save('outputs/specialcase/ols_preds.npy',olspreds)

def save_inds_for_time():
    X_train, X_val, X_test, y_train, y_val, y_test, mol_IDs = prep_data()
    indexes = np.zeros((5),dtype=object)
    for i in range(5):
        indexes[i] = np.vstack([np.arange(0,11000),np.arange(0,11000)]).T
    
    defmaes = np.zeros((4,9),dtype=float)
    olsmaes = np.zeros((4,9),dtype=float)
    modelinds = np.zeros((4),dtype=object)
    
    for fb in range(4):
        for n in range(1,10):
            n_trains = np.asarray([2**(n+4),2**(n+3),2**(n+2),2**(n+1),2**n])[fb:]
            ###TRAINING######
            model = MFML(reg=1e-10, kernel='laplacian', 
                         order=1, metric='l2', 
                         sigma=200.0, p_bar=False)

            model.train(X_train_parent=X_train, 
                        y_trains=y_train[fb:], indexes=indexes[fb:], 
                        shuffle=True, n_trains=n_trains, seed=42)
            #default MFML
            _ = model.predict(X_test = X_test, y_test = y_test, 
                                    X_val = X_val, y_val = y_val, 
                                    optimiser='default')
            defmaes[fb,n-1] = np.copy(model.mae)
            #OLS o-MFML
            _ = model.predict(X_test = X_test, y_test = y_test, 
                                     X_val = X_val, y_val = y_val, 
                                     optimiser='OLS', copy_X= True, 
                                     fit_intercept= False)
            olsmaes[fb,n-1]=np.copy(model.mae)
        modelinds[fb] = np.copy(model.indexes)
    
    np.save('outputs/specialcase/index_def_maes.npy',defmaes)
    np.save('outputs/specialcase/index_ols_maes.npy',olsmaes)
    np.save('outputs/specialcase/index_index.npy',modelinds)
    

main()
predict_main()
save_inds_for_time()
