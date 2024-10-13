# QeMFi: A Multifidelity Dataset of Quantum Chemical Properties of Diverse Molecules
This code repository is connected to the QeMFi dataset and its application. It contains codes to run ORCA calculations for the fidelities, the corresponding input files, and the python scripts to perform various multifidelity calculations as noted in [this preprint](https://arxiv.org/abs/2406.14149). 

The package for python scripts can be installed by cloning this repository and installing the required packages. This can be performed within a new conda environment, say `QeMFi_env`, as follows:

```bash
$ conda create --name QeMFi_env python=3.10.12
``` 

Then, we install the required python libraries with:

```bash
$ conda activate QeMFi_env
(QeMFi_env)$ pip install -r requirements.txt
```

We are now ready to perform ML and MFML for QC with this code repository.

## Generating Molecular Descriptors/Representations
Once the data files are downloaded from the [data repository](https://zenodo.org/records/11636903), one can use them to generate molecular descriptors for ML and Multifidelity ML (MFML). In this code repository, the scripts to generate Coloumb Matrices (CM) and the spectrum of London and Axilrod–Teller–Muto (SLATM) representation are provided. The following example will demonstrate generating the unsorted CMs for nitrophenol from the QeMFi dataset.

```bash
$ conda activate QeMFi_env
(QeMFi_env)$ python GenerateCM.py -m='nitrophenol' -d='path_to_npz_file/' -s='unsorted'
```

The same script can be used to generate row-norm sorted CMs with `-s='row-norm'`. The directory path is the location of the dowloaded data files. The representations will be saved in the current working directory.

One can similarly generate the SLATM representation. In this example, for urea:

```bash
$ conda activate QeMFi_env
(QeMFi_env)$ python GenerateSLATM.py -m='urea' -d='path_to_npz_file/'
```

## Loading The Dataset for use
When working with the QeMFi dataset, once can simply load the data for each molecule using elementary NumPy commands. Some examples given below should be a good starting point for use of the dataset using Python:

```python
import numpy as np

#load the dataset
acrolein_data = np.load('QeMFi_acrolein.npz',allow_pickle=True) #pickled since object array

#list various files in the data
print(acrolein_data.files) #results in ['ID','R','Z','CONF','SCF','EV','TrDP','fosc','DPe','DPn','RCo','DPRo']

​#access oscillator strength values of first excitation state with STO3G fidelity
fosc_STO3G_0 = acrolein_data['fosc'][:,0,0]

#for second with SVP
fosc_SVP_1 = acrolein_data['fosc'][:,3,1]

#access conformation data
confs = acrolein_data['CONF']

```

There are various QC properties provided in this dataset for 5 different fidelities, each of which can be accessed with their appropriate key and array ID. For each molecule there are 15,000 entries for each property at each fidelity.

## Performing Preliminary Multifidelity Analysis
MFML is a powerful method to learn QC properties. In this work package, the ML method of choice is Kernel Ridge Regression (KRR). With KRR, MFML and optimized MFML (o-MFML) are implemented through the scripts. But before the models are implemented, a preliminary analysis of the multifidelity data structure is recommended to anticipate results of MFML (see [Vinod et al. 2023](https://pubs.acs.org/doi/10.1021/acs.jctc.3c00882)). The following example performs the preliminary mutlifidelity analysis and returns the corresponding plots for the x-component of nuclear contribution of molecular dipole moments for alanine:

```bash
$ conda activate QeMFi_env
(QeMFi_env)$ python PrelimAnalysis.py -m='alanine' -d='path_to_npz_file/' -p='DPn' -u='a.u.' -c=0 --centeroffset --saveplot 
```

Simiarly, one can perform the preliminary analysis for the other QC properties. One can get more details about the python script by using `$ python PrelimAnalysis.py --help`.

## MFML Learning Curves
Learning curves indicate the model error (such as MAE or RMSE) wityh increasing model complexity. In the case of KRR, model complexity is controlled by the number of the training samples used. Therefore, one can study the learning curves as MAE vs training samples used. For multifidelity models, the number of training samples used at the highest fidelity are considered to maintain uniform comparison (see [Vinod et al. 2023](https://pubs.acs.org/doi/10.1021/acs.jctc.3c00882) and [Vinod et al. 2024](https://iopscience.iop.org/article/10.1088/2632-2153/ad2cef) for more details on deciphering learning curves).
The following example, generates learning curves for the SCF property of acrolein. Note that the representation of interest should have already been generated (see above).
```bash
$ conda activate QeMFi_env
(QeMFi_env)$ python LearningCurves.py -m='acrolein' -d='path_to_npz_file' -p='SCF' -n=1 -w=150.0 -rep='SLATM' -k='laplacian' -r=1e-10 -s=42 --centeroffset
(QeMFi_env)$ python LC_plots.py -m='acrolein' -p='SCF' -u='hE' -rep='SLATM' --centeroffset --saveplot
```
In addition to the usual learning curves which plot MAE (or some other error) vs the number of trianing samples, the `TimeLC_plots.py` script can be used to generate the plot of time to generate a training set versus MAE (see [Vinod et al. 2023](https://pubs.acs.org/doi/10.1021/acs.jctc.3c00882) for more details). This is achieved with a call similar to the `LC_plots.py` script after running the script to generate the data for learning curves:
```bash
(QeMFi_env)$ python TimeLC_plots.py -m='acrolein' -d='path_to_npzfile/' -p='SCF' -u='hE' -rep='SLATM' --centeroffset --saveplot
```
where the additional `-d` flag corresponds to the directory of the QeMFi dataset to load the data of time calculations.

For more details about each flag, use `$python LearningCurves.py --help`. Note that running the learning curves will take some time and depends also on the number of runs you wish to average over.
The plots are saved as a pdf file.

## Composite Use of QeMFi
In paperURL (TBA), the use of QeMFi as a composite dataset is described. The corresponding script is presented in this code repository as `SpecialStudy.py`. The `prep_data()` function within this script can be modified to generate the composite data set for a desired property. 


## ORCA ipnut file and related scripts
In interest of full transparency, the ORCA calculations scripts and input files for the five fidelities are provided in this code repository. The `ORCA_Calc_job.sh` can be used to run ORCA calculations for a given molecule for a given fidelity. The number of geometries to consider can be modified therein. The QC properties resulting from ORCA calculations can be extracted using the `prop_extraction.sh` script. The text file `single_mol_config.txt` is a lookup table that matches the sequence number (of an ORCA calculation when run in a loop) to the corresponding geometry number from the original WS22 database. The dataset itself was created using the script `CreateDataset.py` which is also provided in this code repository.

## Citing This Work
When using the QeMFi dataset or the scripts provided herein, please cite the following:

1. Vinod, V., & Zaspel, P. (2024). QeMFi: A Multifidelity Dataset of Quantum Chemical Properties of Diverse Molecules (1.1.0) [Data set]. Zenodo. https://doi.org/10.5281/zenodo.13925688
2. **Preprint Version:** Vinod, V., & Zaspel, P. (2024). QeMFi: A Multifidelity Dataset of Quantum Chemical Properties of Diverse Molecules. arXiv preprint arXiv:2406.14149

