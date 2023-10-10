#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  3 16:29:37 2021

@author: dycomp
"""

import pandas as pd
import numpy as np
import pandas as pd
import gensim
from gensim.models import Word2Vec
import rdkit
from rdkit import Chem
import re
import tqdm
import os
os.chdir('F:\\DTI\\CPIScore\\model')

lg = rdkit.RDLogger.logger() 
lg.setLevel(rdkit.RDLogger.CRITICAL)

'''
data = pd.read_excel('./data/total_data.xlsx')
protein_seq = data.prot
drug_smi = data.smiles
word_vec = pd.DataFrame()
dictionary = []
index = []
texts = [[word for word in re.findall(r'.{3}',str(document))] for document in list(protein_seq)]
model = Word2Vec(texts,vector_size=512,window=16,min_count=1,negative=15,sg=1,sample=0.001,hs=1,workers=4)  #12,15
model.save('./result/gensim-model-512dim-protein')
#new_model = gensim.models.Word2Vec.load('gensim-model-32dim-protein')
#print(new_model)
#vectors = pd.DataFrame([new_model.wv.get_vector(word) for word in (new_model.wv.key_to_index)])
#vectors['Word'] = list(new_model.wv.key_to_index)
#print(vectors)

chembl = open('./data/chembl.smi').readlines()
smis = []
for idx,smi in enumerate(chembl):
    mol = Chem.MolFromSmiles(smi)
    if mol is not None and mol.GetNumAtoms() < 60:
        smis.append(smi)
    if idx % 10000 == 0:
        print(idx)
word_vec = pd.DataFrame()
dictionary = []
index = []
texts = [[word for word in re.findall(r'.{3}',str(document))] for document in smis]
model = Word2Vec(texts,vector_size=128,window=16,min_count=1,negative=15,sg=1,sample=0.001,hs=1,workers=32)
model.save('./result/gensim-model-128dim-smiles')
#new_model = gensim.models.Word2Vec.load('gensim-model-256dim-smiles')
#print(new_model)
#vectors = pd.DataFrame([new_model.wv.get_vector(word) for word in (new_model.wv.key_to_index)])
#vectors['Word'] = list(new_model.wv.key_to_index)
#print(vectors)
'''
    
def smiles2vec(smis,dims,window_size,negative_size):
    word_vec = pd.DataFrame()
    dictionary = []
    index = []
    texts = [[word for word in re.findall(r'.{3}',str(document))] for document in smis]
    new_model = gensim.models.Word2Vec.load('./result/gensim-model-256dim-smiles')
    vectors = pd.DataFrame([new_model.wv.get_vector(word) for word in (new_model.wv.key_to_index)])
    vectors['Word'] = list(new_model.wv.key_to_index)
    for i in range(len(smis)):
        index.append(i)
    for i in range(len(texts)):
        i_word = []
        for w in range(len(texts[i])):
            i_word.append(index[i])
        dictionary.extend(i_word)
    word_vec['Id'] = dictionary
        
    dictionary = []
    for i in range(len(texts)):
        i_word = []
        for w in range(len(texts[i])):
            i_word.append(texts[i][w])
        dictionary.extend(i_word)
    word_vec['Word'] = dictionary
    del dictionary,i_word
    word_vec = word_vec.merge(vectors,on='Word',how='left')
    word_vec.columns = ['Id']+['word']+['vec_{0}'.format(i) for i in range(0,dims)]
    return word_vec
    
def feature_embeddings_smiles(smis,dims):
    smiles_vec = smiles2vec(smis,dims,16,15)
    smiles_vec = smiles_vec.drop('word',axis=1)
    name = ['vec_{0}'.format(i) for i in range(0,dims)]
    feature_embeddings = pd.DataFrame(smiles_vec.groupby(['Id'])[name].agg('mean')).reset_index()
    feature_embeddings.columns = ['index']+['mean_ci_{0}'.format(i) for i in range(0,dims)]
    return feature_embeddings


def protein2vec(protein_seq,dims,window_size,negative_size):
    word_vec = pd.DataFrame()
    dictionary = []
    index = []
    texts = [[word for word in re.findall(r'.{3}',str(document))] for document in list(protein_seq)]
    new_model = gensim.models.Word2Vec.load('./result/gensim-model-512dim-protein')
  #  print(new_model)
    vectors = pd.DataFrame([new_model.wv.get_vector(word) for word in (new_model.wv.key_to_index)])
    vectors['Word'] = list(new_model.wv.key_to_index)
  #  print(vectors)
    for i in range(len(protein_seq)):
        index.append(i)
    for i in range(len(texts)):
        i_word = []
        for w in range(len(texts[i])):
            i_word.append(index[i])
        dictionary.extend(i_word)
    word_vec['Id'] = dictionary
        
    dictionary = []
    for i in range(len(texts)):
        i_word = []
        for w in range(len(texts[i])):
            i_word.append(texts[i][w])
        dictionary.extend(i_word)
    word_vec['Word'] = dictionary
    del dictionary,i_word
    word_vec = word_vec.merge(vectors,on='Word',how='left')
    word_vec.columns = ['Id']+['word']+['vec_{0}'.format(i) for i in range(0,dims)]
    return word_vec

def feature_embeddings_protein(protein_seq,dims):
    protein_vec = protein2vec(protein_seq,dims,16,15)
    protein_vec = protein_vec.drop('word',axis=1)
    name = ['vec_{0}'.format(i) for i in range(0,dims)]
    feature_embeddings = pd.DataFrame(protein_vec.groupby(['Id'])[name].agg('mean')).reset_index()
    feature_embeddings.columns = ['index']+['mean_ci_{0}'.format(i) for i in range(0,dims)]
    return feature_embeddings

'''
if __name__ == '__main__':
    print('Molecular Structure and Protein Sequence Continuous Representation')
    try:
        prot_embeddings = feature_embeddings_protein(32)
        drug_embeddings = feature_embeddings_smiles(32)
        prot_embeddings['proteinseq'] = protein_seq
        drug_embeddings['smiles'] = smis
        prot_embeddings.to_csv('train-32dim-protein.csv',index=False,sep='\t')
        drug_embeddings.to_csv('train-32dim-drug.csv',index=False,sep='\t')
    except ImportError:
        print('Error!')
'''




























