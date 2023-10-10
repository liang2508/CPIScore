# -*- coding: utf-8 -*-
"""
Created on Tue Aug  4 15:10:32 2020

@author: Administrator
"""

import numpy as np
import pandas as pd
import time
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error,r2_score,mean_absolute_error
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold,mutual_info_regression
import matplotlib.pyplot as plt
from rdkit import Chem
from rdkit.Chem.rdmolops import GetAdjacencyMatrix
from rdkit.Chem import rdPartialCharges
import scipy.sparse as sp
from rdkit.Chem import AllChem
from rdkit.Chem import Descriptors
from rdkit.ML.Descriptors import MoleculeDescriptors
from rdkit.Chem.Descriptors import rdMolDescriptors

def finger(mol):
    fps = []
    for m in mol:
        fp = AllChem.GetMorganFingerprintAsBitVect(m,2,nBits=2048)
        fp = fp.ToBitString()
        fp = np.array(list(fp)).astype('int8')
        fps.append(fp)
    return np.array(fps)
    

def rdkit2D(mol):
    descs = [desc_name[0] for desc_name in Descriptors._descList]
    calc = MoleculeDescriptors.MolecularDescriptorCalculator(descs)
    descriptors = []
    for m in mol:
        des = calc.CalcDescriptors(m)
        descriptors.append(des)
    return np.array(descriptors)


def rdkit3D(mol):
    descriptors = []
    for m in mol:
        descriptor = []
        descriptor.extend(rdMolDescriptors.CalcAUTOCORR3D(m))
        descriptor.extend(rdMolDescriptors.CalcMORSE(m))
        descriptor.extend(rdMolDescriptors.CalcRDF(m))
        descriptor.extend(rdMolDescriptors.CalcWHIM(m))
        descriptors.append(descriptor)
    return np.array(descriptors)


def TransNan(dataset):
    for i in range(dataset.shape[1]):
        mean_des = np.nanmean(dataset[:,i])
        for j in range(dataset.shape[0]):
            if np.isnan(dataset[j,i]):
                dataset[j,i] = mean_des
    return dataset

def normalize_adj(adj):
    row = []
    for i in range(adj.shape[0]):
        sum = adj[i].sum()
        row.append(sum)
    rowsum = np.array(row)
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    a = d_mat_inv_sqrt.dot(adj)
    return a
        

def one_hot(x,allowable_set):
    if x not in allowable_set:
        raise Exception('input {0} not in allowable set{1}'.format(x,allowable_set))
    return list(map(lambda s:x==s,allowable_set))
        
def one_hot_pad(x,allowable_set):
    if x not in allowable_set:
        x = allowable_set[-1]
    return list(map(lambda s:x==s,allowable_set))
        
def NullToZero(x):
    if np.isnan(x) or np.isinf(x):
        x = 0
    else:
        x = x
    return x
           
def atom_feature(m,atom_i):
    atom = m.GetAtomWithIdx(atom_i)
    Chem.rdPartialCharges.ComputeGasteigerCharges(m)
    return np.array(one_hot_pad(atom.GetSymbol(),['C','N','O','S','P','B','F','Cl','Br','I','H'])+
                    one_hot(atom.GetDegree(),[0,1,2,3,4,5])+
                    one_hot(atom.GetTotalNumHs(),[0,1,2,3,4])+
                    one_hot(atom.GetImplicitValence(),[0,1,2,3,4,5])+
                    [atom.GetIsAromatic()]+[atom.IsInRing()]+[NullToZero(atom.GetDoubleProp('_GasteigerCharge'))])


def cal_rdkit(train,val,test):
    D_train = rdkit2D(train)
    D_train = TransNan(D_train)
    st = StandardScaler()
    D_train_St = st.fit_transform(D_train)
    D_val = rdkit2D(val)
    D_val = TransNan(D_val)
    D_val_St = st.transform(D_val)
    D_test = rdkit2D(test)
    D_test = TransNan(D_test)
    D_test_St = st.transform(D_test)
    return D_train_St, D_val_St, D_test_St

def cal_rdkit3D(train,val,test):
    D_train = rdkit3D(train)
    D_train = TransNan(D_train)
    D_val = rdkit3D(val)
    D_val = TransNan(D_val)
    D_test = rdkit3D(test)
    D_test = TransNan(D_test)
    return D_train, D_val, D_test
    
def cal_finger(train,val,test):
    D_train = finger(train)
    D_val = finger(val)
    D_test = finger(test)
    return D_train, D_val, D_test


def cal_atomprop(train,val,test):
    max_natoms = 50
    train_X = []
    val_X = []
    test_X = []
    for i in range(len(train)):
        m = train[i]
        natoms = m.GetNumAtoms()
        train_x = [atom_feature(m,i) for i in range(natoms)]
        for i in range(natoms,max_natoms):
            train_x.append(np.zeros(31))
        train_X.append(train_x)
    for i in range(len(val)):
        m = val[i]
        natoms = m.GetNumAtoms()
        val_x = [atom_feature(m,i) for i in range(natoms)]
        for i in range(natoms,max_natoms):
            val_x.append(np.zeros(31))
        val_X.append(val_x)    
    for i in range(len(test)):
        m = test[i]
        natoms = m.GetNumAtoms()
        test_x = [atom_feature(m,i) for i in range(natoms)]
        for i in range(natoms,max_natoms):
            test_x.append(np.zeros(31))
        test_X.append(test_x)
    return np.array(train_X), np.array(val_X), np.array(test_X)

def cal_adjacentmatrix(train,val,test):
    max_natoms = 50
    train_A = []
    val_A = []
    test_A = []    
    for i in range(len(train)):
        m = train[i]
        natoms = m.GetNumAtoms()
        train_a = GetAdjacencyMatrix(m)+np.eye(natoms)
        train_a_norm = normalize_adj(train_a)
        train_a_padding = np.zeros((max_natoms,max_natoms))
        train_a_padding[:natoms,:natoms] = train_a_norm
        train_a_padding[:natoms,:natoms] = train_a
        train_A.append(train_a_padding)
    for i in range(len(val)):
        m = val[i]
        natoms = m.GetNumAtoms()
        val_a = GetAdjacencyMatrix(m)+np.eye(natoms)
        val_a_norm = normalize_adj(val_a)
        val_a_padding = np.zeros((max_natoms,max_natoms))
        val_a_padding[:natoms,:natoms] = val_a_norm
        val_a_padding[:natoms,:natoms] = val_a
        val_A.append(val_a_padding)
    for i in range(len(test)):
        m = test[i]
        natoms = m.GetNumAtoms()
        test_a = GetAdjacencyMatrix(m)+np.eye(natoms)
        test_a_norm = normalize_adj(test_a)
        test_a_padding = np.zeros((max_natoms,max_natoms))
        test_a_padding[:natoms,:natoms] = test_a_norm
        test_a_padding[:natoms,:natoms] = test_a
        test_A.append(test_a_padding)
    return np.array(train_A), np.array(val_A), np.array(test_A)

'''
def Mutual_info(x,y):                  #去掉百分之十最不相关的特征
    mi = mutual_info_regression(x,y)
    threshold = {}
    for i in range(len(list(mi))):
        threshold[i] = list(mi)[i]
    percen_value = int(0.1 * len(list(mi)))
    threshold_value = sorted(threshold.values(),reverse=False)[percen_value]
    index = [i for i in threshold.keys() if threshold[i] <= threshold_value] 
    x = np.delete(x,index,axis=1)
    return x
'''
def Mutual_info(x,y):                  #去掉百分之十最不相关的特征
    mi = mutual_info_regression(x,y)
    threshold = {}
    for i in range(len(list(mi))):
        threshold[i] = list(mi)[i]
    threshold = sorted(threshold.items(),key=lambda item:item[1])
    percen_value = int(0.1 * len(list(mi)))
    threshold = threshold[:percen_value]
    index = [i[0] for i in threshold] 
    x = np.delete(x,index,axis=1)
    return x


  

