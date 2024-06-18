import numpy as np

def create_CheMFiDataset(molname):
    fids = np.asarray(['STO3G','321G','631G','SVP','TZVP'])
    sampling = np.arange(0,120000)[::int(120000/15000)]
    
    ws22_raw = np.load(f'ws22_{molname}.npz',allow_pickle=True) #raw file from WS22 database
    coords = ws22_raw['R'][sampling]
    Zs = ws22_raw['Z']
    confs = ws22_raw['CONF'][:,0][sampling]
    
    SCF = np.zeros((15000,5),dtype=float)
    EV = np.zeros((15000,5,10),dtype=float)
    TrDP = np.zeros((15000,5,10,3),dtype=float)
    fosc = np.zeros((15000,5,10),dtype=float)
    DPe = np.zeros((15000,5,3),dtype=float)
    DPn = np.zeros((15000,5,3),dtype=float)
    RCo = np.zeros((15000,5,3),dtype=float)
    DPRo = np.zeros((15000,5,3),dtype=float)
    
    for i in range(5):
        SCF[:,i] = np.loadtxt(f'propertyzips/{molname}/{fids[i]}_SCF.dat') #*27.211407953 #convert eh to ev
        EV[:,i,:] = np.loadtxt(f'propertyzips/{molname}/{fids[i]}_EV.dat') #*0.000124 #convert cm-1 to ev
        TrDP[:,i,:,0] = np.loadtxt(f'propertyzips/{molname}/{fids[i]}_TX.dat')
        TrDP[:,i,:,1] = np.loadtxt(f'propertyzips/{molname}/{fids[i]}_TY.dat')
        TrDP[:,i,:,2] = np.loadtxt(f'propertyzips/{molname}/{fids[i]}_TZ.dat')
        fosc[:,i,:] = np.loadtxt(f'propertyzips/{molname}/{fids[i]}_fosc.dat')
        DPe[:,i,:] = np.loadtxt(f'propertyzips/{molname}/{fids[i]}_DPe.dat')
        DPn[:,i,:] = np.loadtxt(f'propertyzips/{molname}/{fids[i]}_DPn.dat')
        RCo[:,i,:] = np.loadtxt(f'propertyzips/{molname}/{fids[i]}_RotConst.dat')
        DPRo[:,i,:] = np.loadtxt(f'propertyzips/{molname}/{fids[i]}_DPRo.dat')
    
    np.savez(f'dataset/CheMFi_{molname}.npz',
             ID=sampling, R=coords, Z=Zs, CONF=confs,
             SCF=SCF, 
             EV=EV, TrDP=TrDP, fosc=fosc,
             DPe=DPe, DPn=DPn, 
             RCo=RCo, DPRo=DPRo
            )
    
def main():
    mol_list = np.asarray(['urea','acrolein','alanine','sma','nitrophenol','urocanic','dmabn','thymine','o-hbdi'])
    for m in mol_list:
        create_CheMFiDataset(molname=m)

main()
