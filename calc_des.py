# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.ML.Descriptors import MoleculeDescriptors
from sklearn.preprocessing import StandardScaler
import math
import re
import os
os.chdir('F:\\DTI\\CPIScore\\model')
import w2v

train_set = np.array(pd.read_csv('./data/train_set.csv',header=0))
test_set = np.array(pd.read_csv('./data/test_set.csv',header=0))

train_proseq = train_set[:,1]
test_proseq = test_set[:,1]
train_smi = train_set[:,2]
test_smi = test_set[:,2]
total_smi = np.concatenate([train_smi,test_smi])
train_prop = train_set[:,3]
test_prop = test_set[:,3]

pdb_train = train_set[:,0]
pdb_test = test_set[:,0]

smivec_train = np.array(w2v.feature_embeddings_smiles(train_smi,256).drop('index',axis=1))
smivec_test = np.array(w2v.feature_embeddings_smiles(test_smi,256).drop('index',axis=1))
provec_train = np.array(w2v.feature_embeddings_protein(train_proseq,512).drop('index',axis=1))
provec_test = np.array(w2v.feature_embeddings_protein(test_proseq,512).drop('index',axis=1))


def lig_index(charset,smis):
    char_to_int = dict((c,i) for i,c in enumerate(charset))
    int_to_char = dict((i,c) for i,c in enumerate(charset))
    smis_new = []
    for smi in smis:
        while len(smi) < 256:
            smi = smi + 'E'
        smis_new.append(smi)
    words_lig = [[word for word in re.findall(r'.{1}',str(document))] for document in smis_new]
    words_indexs_list_lig = []
    for i in range(len(words_lig)):
        word = words_lig[i]
        index = []
        index = [int(char_to_int[char]) for char in word]
        words_indexs_list_lig.append(index)
    words_indexs_lig = np.array(words_indexs_list_lig)
    return words_indexs_lig

def pro_index(seqset,seqs):
    char_to_int = dict((c,i) for i,c in enumerate(seqset))
    int_to_char = dict((i,c) for i,c in enumerate(seqset))
    '''
    seqs_new = []
    for seq in seqs:
        if 'X' in seq:
            seq = seq.replace('X','!')
        while len(seq) < 512:
            seq = seq + '!'
        seqs_new.append(seq)
    words_pro = [[word for word in re.findall(r'.{1}',str(document)) if word in seqset] for document in seqs_new]
    '''
    seqs_new = [[word for word in re.findall(r'.{1}',str(document))] for document in seqs]
    words_pro = []
    for seq in seqs_new:
        for i in range(len(seq)):
            if seq[i] not in seqset:
                seq[i] = '!'
        while len(seq) < 512:
            seq.append('!')
        words_pro.append(seq)
    words_indexs_list_pro = []
    for i in range(len(words_pro)):
        word = words_pro[i]
        index = []
        index = [int(char_to_int[char]) for char in word]
        words_indexs_list_pro.append(index)
    words_indexs_pro = np.array(words_indexs_list_pro)
    return words_indexs_pro

charset = ['[','N','H','3','+',']','C','@','(','=','O',')','S','c','1','-','2','l','n','P','F','4','s','\\',
           '/','5',':','6','7','8','I','B','r','o','#','.','9','E']
smiindex_train = lig_index(charset,train_smi)
smiindex_test = lig_index(charset,test_smi)

seqset = ['G','A','V','L','I','P','F','Y','W','S','T','C','M','N','Q','K','R','H','D','E','!']
seqindex_train = pro_index(seqset,train_proseq)
seqindex_test = pro_index(seqset,test_proseq)

def rdkit2D(mols):
    descs = [desc_name[0] for desc_name in Descriptors._descList]
    calc = MoleculeDescriptors.MolecularDescriptorCalculator(descs)
    descriptors = []
    for mol in mols:
        des = calc.CalcDescriptors(mol)
        descriptors.append(des)
    return np.array(descriptors)

def TransNan(data_set):
    for i in range(data_set.shape[1]):
        mean_des = np.nanmean(data_set[:,i])
        for j in range(data_set.shape[0]):
            if np.isnan(data_set[j,i]):
                data_set[j,i] = mean_des
    return data_set

train_mol = [Chem.MolFromSmiles(smi) for smi in train_smi]
test_mol = [Chem.MolFromSmiles(smi) for smi in test_smi]
train_des = rdkit2D(train_mol)
test_des  = rdkit2D(test_mol)
train_des = TransNan(train_des)
test_des = TransNan(test_des)

total_des = np.concatenate([train_des,test_des])
total_prop = np.concatenate([train_prop,test_prop])
stlig = StandardScaler()
train_des = stlig.fit_transform(train_des)
test_des = stlig.transform(test_des)
