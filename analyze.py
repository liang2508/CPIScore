# -*- coding: utf-8 -*-
"""
Created on Sun Jul 17 22:19:13 2022

@author: Administrator
"""
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import rdkit
from rdkit import Chem
from rdkit.Chem import AllChem,Descriptors
from rdkit.ML.Descriptors import MoleculeDescriptors
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import os
os.chdir('/public/home/cadd1/liang/DTI/newmodel')

lg = rdkit.RDLogger.logger() 
lg.setLevel(rdkit.RDLogger.CRITICAL)

def calc_prop(smis):
    props = []
    for smi in tqdm(smis):
        mol = Chem.MolFromSmiles(smi)
        mw = Descriptors.MolWt(mol)
        hbd = Chem.Lipinski.NumHDonors(mol)
        hba = Chem.Lipinski.NumHAcceptors(mol)
        rb = Chem.Lipinski.NumRotatableBonds(mol)
        clogp = Descriptors.MolLogP(mol)
        tpsa = Descriptors.TPSA(mol)
        props.append([mw,hbd,hba,rb,clogp,tpsa])
    return props

def calc_fp(smis):
    mols = [Chem.MolFromSmiles(smi) for smi in smis]
    fps = []
    for m in tqdm(mols):
        fp = AllChem.GetMorganFingerprintAsBitVect(m, 2, 1024)
        fp = fp.ToBitString()
        fp = np.array(list(fp)).astype('int8')
        fps.append(list(fp))
    return fps

def reduce_pca(propslist):
    data = PCA(n_components = 2)
    decomp = data.fit_transform(np.array(propslist))
    x = decomp[:,0]
    y = decomp[:,1]
    return x,y

def reduce_tsne(propslist):
    data = TSNE(n_components = 2)
    decomp = data.fit_transform(np.array(propslist))
    x = decomp[:,0]
    y = decomp[:,1]
    return x,y

def sim_fp(smi1,smilist):
    mol1 = Chem.MolFromSmiles(smi1)
    fp1 = AllChem.GetMorganFingerprint(mol1,2)
    mols = [Chem.MolFromSmiles(smi) for smi in smilist]
    fps = [AllChem.GetMorganFingerprint(mol,2) for mol in mols]
    sims = [Chem.DataStructs.TanimotoSimilarity(fp1,fp2) for fp2 in fps]
    return np.max(sims)

train_smis = list(pd.read_csv('./train_set.csv').iloc[:,2])
test_smis = list(pd.read_csv('./test_set.csv').iloc[:,2])
train_prop, test_prop = calc_prop(train_smis), calc_prop(test_smis)
train_fp, test_fp = calc_fp(train_smis), calc_fp(test_smis)

# pca reduce dimension
x_train_prop_pca, y_train_prop_pca = reduce_pca(train_prop)
x_test_prop_pca, y_test_prop_pca = reduce_pca(test_prop)
train_prop_pca = pd.DataFrame([x_train_prop_pca,y_train_prop_pca],index=['PC1','PC2']).T
test_prop_pca = pd.DataFrame([x_test_prop_pca, y_test_prop_pca],index=['PC1','PC2']).T
train_prop_pca.to_csv('train_prop_pca.csv',index=None)
test_prop_pca.to_csv('test_prop_pca.csv',index=None)
x_train_fp_pca, y_train_fp_pca = reduce_pca(train_fp)
x_test_fp_pca, y_test_fp_pca = reduce_pca(test_fp)
train_fp_pca = pd.DataFrame([x_train_fp_pca,y_train_fp_pca],index=['PC1','PC2']).T
test_fp_pca = pd.DataFrame([x_test_fp_pca, y_test_fp_pca],index=['PC1','PC2']).T
train_fp_pca.to_csv('train_fp_pca.csv',index=None)
test_fp_pca.to_csv('test_fp_pca.csv',index=None)

# tsne reduce dimension
x_train_prop_tsne, y_train_prop_tsne = reduce_tsne(train_prop)
x_test_prop_tsne, y_test_prop_tsne = reduce_tsne(test_prop)
train_prop_tsne = pd.DataFrame([x_train_prop_tsne, y_train_prop_tsne],index=['t-SNE1','t-SNE2']).T
test_prop_tsne = pd.DataFrame([x_test_prop_tsne, y_test_prop_tsne],index=['t-SNE1','t-SNE2']).T
train_prop_tsne.to_csv('train_prop_tsne.csv',index=None)
test_prop_tsne.to_csv('test_prop_tsne.csv',index=None)
x_train_fp_tsne, y_train_fp_tsne = reduce_tsne(train_fp)
x_test_fp_tsne, y_test_fp_tsne = reduce_tsne(test_fp)
train_fp_tsne = pd.DataFrame([x_train_fp_tsne, y_train_fp_tsne],index=['t-SNE1','t-SNE2']).T
test_fp_tsne = pd.DataFrame([x_test_fp_tsne, y_test_fp_tsne],index=['t-SNE1','t-SNE2']).T
train_fp_tsne.to_csv('train_fp_tsne.csv',index=None)
test_fp_tsne.to_csv('test_fp_tsne.csv',index=None)




# draw chemical space distribution plot
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
os.chdir('/public/home/cadd1/liang/DTI/newmodel')

train_prop_pca, test_prop_pca = pd.read_csv('train_prop_pca.csv'), pd.read_csv('test_prop_pca.csv')
train_prop_pca['Data'], test_prop_pca['Data'] = 'train', 'test'
train_fp_pca, test_fp_pca = pd.read_csv('train_fp_pca.csv'), pd.read_csv('test_fp_pca.csv')
train_fp_pca['Data'], test_fp_pca['Data'] = 'train', 'test'

train_prop_tsne, test_prop_tsne = pd.read_csv('train_prop_tsne.csv'), pd.read_csv('test_prop_tsne.csv')
train_prop_tsne['Data'], test_prop_tsne['Data'] = 'train', 'test'
train_fp_tsne, test_fp_tsne = pd.read_csv('train_fp_tsne.csv'), pd.read_csv('test_fp_tsne.csv')
train_fp_tsne['Data'], test_fp_tsne['Data'] = 'train', 'test'

pca_prop = pd.concat([train_prop_pca,test_prop_pca])
pca_fp = pd.concat([train_fp_pca,test_fp_pca])
tsne_prop = pd.concat([train_prop_tsne,test_prop_tsne])
tsne_fp = pd.concat([train_fp_tsne,test_fp_tsne])
fig,ax = plt.subplots(2, 2, figsize=(12, 12))
g1 = sns.scatterplot(x='PC1',y='PC2',data=pca_prop,hue='Data',palette='husl',ax=ax[0][0])
g2 = sns.scatterplot(x='PC1',y='PC2',data=pca_fp,hue='Data',palette='husl',ax=ax[0][1])
g3 = sns.scatterplot(x='t-SNE1',y='t-SNE2',data=tsne_prop,hue='Data',palette='husl',ax=ax[1][0])
g4 = sns.scatterplot(x='t-SNE1',y='t-SNE2',data=tsne_fp,hue='Data',palette='husl',ax=ax[1][1])
#g1.legend(fontsize=14,loc='upper right',borderpad=0.1,borderaxespad=0.1)
#g2.legend(fontsize=14,loc='upper right',borderpad=0.1,borderaxespad=0.1)
#g3.legend(fontsize=14,loc='upper right',borderpad=0.1,borderaxespad=0.1)
#g4.legend(fontsize=14,loc='upper right',borderpad=0.1,borderaxespad=0.1)
#g.legend([],[], frameon=False)   #移除图例
sns.despine()      #移除顶部和右侧坐标轴
sns.set_style('darkgrid')
for ax in plt.gcf().axes:
    l = ax.get_xlabel()
    ll = ax.get_ylabel()
    ax.set_xlabel(l, fontsize=14)
    ax.set_ylabel(ll,fontsize=14)
    h,l = ax.get_legend_handles_labels()   # get handles and labels from the data so you can edit them
    ax.legend(handles=h,labels=['train', 'test'],fontsize=14,loc='upper right')
plt.savefig('reduce.png', dpi=200, bbox_inches='tight')
plt.show()

'''
# violin graph to observe affinity value distribution
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
y_train = list(pd.read_csv('train_set.csv').iloc[:,3])
y_test = list(pd.read_csv('test_set.csv').iloc[:,3])
y = y_train + y_test
label = ['dataset','train_set','test_set']
sns.violinplot(data=[y,y_train,y_test],label=label)
plt.xticks(ticks=[0,1,2],labels=label,fontsize=11)
plt.ylabel('PIC50',{'size':11})
#plt.title('Distribution of affinity values')
plt.savefig('analyze2.png',bbox_inches='tight')
plt.show()
'''