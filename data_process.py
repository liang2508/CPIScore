# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import cv2
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit import DataStructs
from sklearn.feature_selection import SelectKBest,SelectPercentile,VarianceThreshold
from sklearn.feature_selection import f_regression,mutual_info_regression
import math
import os
os.chdir('F:\\DTI\\CPIScore\\model')

# for classification: <=500 nM (>=6.301) label=1, >500 nM (<6.301) label=0

np.random.seed(0)

data = pd.read_excel('./data/total_data.xlsx',index_col=None,header=0)

pdbs = data.iloc[:,0]
dict_pro = {}
dict_smi = {}
prot = data.iloc[:,1]
lig = data.iloc[:,2]

# 删除去有相同分子的pdb
'''
for i in range(len(prot)):
    if prot[i] in dict_pro.keys():
        dict_pro[prot[i]].append(pdbs[i])
    else:
        dict_pro[prot[i]] = [pdbs[i]]
for i in range(len(pdbs)):
    dict_smi[pdbs[i]] = lig[i]
pdbs_same = []
for id_pro in dict_pro.keys():
    names = dict_pro[id_pro]
    smis = []
    name_smi = {}
    if len(names) > 1:
        for name in names:
            smis.append(dict_smi[name])
            name_smi[name] = dict_smi[name]
        smis_same = [smi for smi in smis if smis.count(smi)>1]
        smis_sames = []
        for smi in smis_same:
            if smi not in smis_sames:
                smis_sames.append(smi)
        for smi in smis_sames:
            pdbs_same.append([k for k,v in name_smi.items() if v == smi])
pdb_removed = []
for ele in pdbs_same:
    pdb_removed.extend(ele[1:])
    ilist = data[data.loc[:,'name'].isin(ele)].index
    values = []
    for i in ilist:
        values.append(data.iloc[i,3])
    data.iloc[ilist[0],3] = sum(values)/len(values)
    data.iloc[ilist[0],3] = round(data.iloc[ilist[0],3],2)
for name in pdb_removed:
    data = data[data.name != name] 
'''

pdbs = data.iloc[:,0]
pdb_removed = []
data_core = pd.read_excel('./data/core_data.xlsx',index_col=None,header=0)
for name in list(data.iloc[:,0]):
    if name in list(data_core.iloc[:,0]):
        pdb_removed.append(name)
        
for i in range(data.shape[0]):
    smi = data.iloc[i,2]
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        pdb_removed.append(pdbs[i])
    else:
        natom = mol.GetNumAtoms()
      #  if natom > 60 or natom < 10:
        if natom > 60:
            pdb_removed.append(pdbs[i])

for i in range(data.shape[0]):
    seq = data.iloc[i,1]
    if len(seq) > 512 and pdbs[i] not in pdb_removed:
        pdb_removed.append(pdbs[i])
'''     
for pdb in pdbs:
    try:
        img = cv2.imread(r'/data/liang/ML/total_ligands_png/'+pdb+'.png')
        rows,cols,channel = img.shape
    except:
        pdb_removed.append(pdb)
'''

for name in pdb_removed:
    data = data[data.name != name]
data = np.array(shuffle(data))[:,:]
label = data[:,3]
complex = data[:,:]

data_train,data_test,y_train,y_test = train_test_split(complex,label,test_size=0.2,random_state=2022)

data_train = pd.DataFrame(data_train)
data_test = pd.DataFrame(data_test)  
complex = pd.DataFrame(complex)  

data_train.to_csv('./data/train_set.csv',index=False)
data_test.to_csv('./data/test_set.csv',index=False)
complex.to_csv('./data/dataset.csv',index=False)
print('data process completed')

'''
for i in range(data.shape[0]):
    smi1 = data.iloc[i,2]
    mol1 = Chem.MolFromSmiles(smi1)
    fp1 = AllChem.GetMorganFingerprint(mol1,2)
    for j in range(data.shape[0]):
        if i == j:
            break
        smi2 = data.iloc[j,2]
        mol2 = Chem.MolFromSmiles(smi2)
        fp2 = AllChem.GetMorganFingerprint(mol2,2)
        if DataStructs.DiceSimilarity(fp1,fp2) == 1:
            pdb_removed.append(pdbs[i])
'''