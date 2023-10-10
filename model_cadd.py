#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 21 14:40:32 2021

@author: dycomp
"""

import numpy as np
from sklearn.metrics import r2_score
from scipy.stats import pearsonr
import time
import torch 
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.optim.lr_scheduler as lr_scheduler

import calc_des
from utils import Transformer,MolDataset

np.random.seed(1995)

pdb_train = calc_des.pdb_train
pdb_test = calc_des.pdb_test
train_mol = calc_des.train_mol
test_mol = calc_des.test_mol
y_train = calc_des.train_prop
y_test = calc_des.test_prop

smivec_train = calc_des.smivec_train
smivec_test = calc_des.smivec_test
provec_train = calc_des.provec_train
provec_test = calc_des.provec_test

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.embedding = nn.Linear(37, 128)
        self.W = nn.ModuleList([nn.Linear(128, 128) for layer in range(2)])
        self.conv = nn.Conv1d(60, 60, 5, padding=2)
        self.bn = nn.BatchNorm1d(5)
        self.pool = nn.MaxPool1d(3, stride=1, padding=1)

        self.fc1 = nn.Linear(1152, 128)
        self.fc2 = nn.Linear(128, 10)
        self.fc3 = nn.Linear(10, 1)
        self.drop = nn.Dropout(0.2)

        self.dnn1 = nn.Linear(512,256)
        self.dnn2 = nn.Linear(256,128)
        self.dnn3 = nn.Linear(128,256)
        self.dnn4 = nn.Linear(256,512)
        self.dnn5 = nn.Linear(256,128)
        self.dnn6 = nn.Linear(128,256)

    def layer_vecl(self,vec):
        vec = F.relu(self.dnn5(vec))
        vec = self.drop(vec)
        vec = F.relu(self.dnn6(vec))
        return vec

    def layer_vecp(self,vec):
        vec = F.relu(self.dnn1(vec))
        vec = self.drop(vec)
        vec = F.relu(self.dnn2(vec))
        vec = self.drop(vec)
        vec = F.relu(self.dnn3(vec))
        vec = self.drop(vec)
        vec = F.relu(self.dnn4(vec))
        return vec

    def forward(self, x, A, l, p, vecl, vecp):
        x = self.embedding(x)
        for layernum in self.W:
            x = layernum(x)
            x = A @ x
            residue = x
            x = F.relu(self.conv(x))
            x = self.pool(x)
            x = self.conv(x)
            x = torch.add(x, residue)
            x = F.relu(x)
        x = x.mean(1)

        vecl = self.layer_vecl(vecl)
        vecp = self.layer_vecp(vecp)
        
        embed_size, num_layers, heads, forward_expansion = 256, 4, 4, 0.5
        tfm = Transformer(device, embed_size, num_layers, heads, forward_expansion).to(device)
      #  tfm_out = tfm(p,l)
        tfm_out = tfm(l,p)

        out = torch.cat((x, vecl, vecp, tfm_out), 1)
        try:
            output = self.fc1(out)
        except:
            print('out_shape:', out.shape)
            out = a + b
        output = self.fc2(output)
        output = self.fc3(output)
        return output

def val(model,test_gcnloader,test_ligloader,test_proloader,test_smivecloader,test_provecloader,loss_fn):
    model.eval()
    with torch.no_grad():
        val_loss = []
        for batch in zip(test_gcnloader,test_ligloader,test_proloader,test_smivecloader,test_provecloader):
            x, A, y, l, p, vecl, vecp = batch[0]['X'].float(), batch[0]['A'].float(), batch[0]['Y'].float(), \
                             batch[1], batch[2], batch[3].float(), batch[4].float()
            x, A, y, l, p, vecl, vecp = x.to(device), A.to(device), y.to(device), l.to(device), p.to(device), vecl.to(device), vecp.to(device)
            pred = model(x, A, l, p, vecl, vecp).squeeze(-1)
            loss = loss_fn(pred, y)
            val_loss.append(loss.data.cpu().numpy())
    return np.mean(val_loss)

def model_save(epoch,model,test_gcnloader,test_ligloader,test_proloader,test_smivecloader,test_provecloader):
    model.eval()
    with torch.no_grad():
        true_test,pred_test = [],[]
        for batch in zip(test_gcnloader,test_ligloader,test_proloader,test_smivecloader,test_provecloader):
            x, A, y, l, p, vecl, vecp = batch[0]['X'].float(), batch[0]['A'].float(), batch[0]['Y'].float(), \
                             batch[1], batch[2], batch[3].float(), batch[4].float()
            x, A, y, l, p, vecl, vecp = x.to(device), A.to(device), y.to(device), l.to(device), p.to(device), vecl.to(device), vecp.to(device)
            pred = model(x,A,l,p,vecl,vecp).squeeze(-1)
            pred_test.append(pred.data.cpu().numpy())
            true_test.append(y.data.cpu().numpy())
    pred_test = np.concatenate(pred_test,-1)
    true_test = np.concatenate(true_test,-1)
    test_r = pearsonr(true_test,pred_test)
    if test_r[0] > 0.75:
        print('***************save model*****************')
        print('epoch:',epoch,',test_r:',test_r)
        torch.save(model,'model_'+str(epoch)+'.pkl')

lig_train = calc_des.smiindex_train
lig_test = calc_des.smiindex_test
prot_train = calc_des.seqindex_train
prot_test = calc_des.seqindex_test

y_train = calc_des.train_prop
y_test = calc_des.test_prop

model = Model()
device = torch.device('cuda:3' if torch.cuda.is_available() else 'cpu')
model.to(device)

batch_size = 64
lr = 4e-5

train_ligset = MolDataset(train_mol,y_train,60)
test_ligset = MolDataset(test_mol,y_test,60)
train_gcnloader = DataLoader(train_ligset,batch_size=batch_size,num_workers=0)
test_gcnloader = DataLoader(test_ligset,batch_size=batch_size,num_workers=0)

train_ligloader = DataLoader(lig_train,batch_size=batch_size,num_workers=0)
test_ligloader = DataLoader(lig_test,batch_size=batch_size,num_workers=0)
train_proloader = DataLoader(prot_train,batch_size=batch_size,num_workers=0)
test_proloader = DataLoader(prot_test,batch_size=batch_size,num_workers=0)

train_smivecloader = DataLoader(smivec_train,batch_size=batch_size,num_workers=0)
test_smivecloader = DataLoader(smivec_test,batch_size=batch_size,num_workers=0)
train_provecloader = DataLoader(provec_train,batch_size=batch_size,num_workers=0)
test_provecloader = DataLoader(provec_test,batch_size=batch_size,num_workers=0)

for p in model.parameters():
    if p.dim() > 1:
        nn.init.xavier_uniform_(p)
# this code is very important! It initialises the parameters with a
# range of values that stops the signal fading or getting too big.

loss_fn = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
lambda1 = lambda epoch: 0.99
scheduler = lr_scheduler.MultiplicativeLR(optimizer, lambda1)

start = time.time()
train_loss_epoch = []
val_loss_epoch = []

model_parameters = filter(lambda p: p.requires_grad, model.parameters())
params = sum([np.prod(p.size()) for p in model_parameters])
print(params)

for epoch in range(200):
    model.train()
 #   scheduler.step()
    if epoch == 150 or epoch == 160 or epoch == 180:
        torch.save(model, 'model_' + str(epoch) + '.pkl')
        print('optlr:', optimizer.state_dict()['param_groups'][0]['lr'])
    train_loss = []
    for i_batch,batch in enumerate(zip(train_gcnloader,train_ligloader,train_proloader,train_smivecloader,train_provecloader)):
        x, A, y, l, p, vecl, vecp = batch[0]['X'].float(), batch[0]['A'].float(), batch[0]['Y'].float(), \
                            batch[1], batch[2], batch[3].float(), batch[4].float()
        x, A, y, l, p, vecl, vecp = x.to(device), A.to(device), y.to(device), l.to(device), p.to(device), vecl.to(device), vecp.to(device)
        pred = model(x,A,l,p,vecl,vecp).squeeze(-1)
        loss = loss_fn(pred.float(), y.float())
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(),1.0)
        optimizer.step()     #update weight
        train_loss.append(loss.data.cpu().numpy())
    train_loss_epoch.append(np.mean(train_loss))
    val_loss = val(model,test_gcnloader,test_ligloader,test_proloader,test_smivecloader,test_provecloader,loss_fn)
    val_loss_epoch.append(val_loss)
    model_save(epoch, model, test_gcnloader, test_ligloader, test_proloader, test_smivecloader,test_provecloader)

    import pandas as pd
    from rdkit import Chem
    import w2v
    import re
    def lig_index(smis,
                  charset=['[', 'N', 'H', '3', '+', ']', 'C', '@', '(', '=', 'O', ')', 'S', 'c', '1', '-', '2', 'l',
                           'n', 'P', 'F', '4', 's', '\\', '/', '5', ':', '6', '7', '8', 'I', 'B', 'r', 'o', '#', '.', '9', 'E']):
        char_to_int = dict((c, i) for i, c in enumerate(charset))
        int_to_char = dict((i, c) for i, c in enumerate(charset))
        smis_new = []
        for smi in smis:
            while len(smi) < 256:
                smi = smi + 'E'
            smis_new.append(smi)
        words_lig = [[word for word in re.findall(r'.{1}', str(document))] for document in smis_new]
        words_indexs_list_lig = []
        for i in range(len(words_lig)):
            word = words_lig[i]
            index = []
            index = [int(char_to_int[char]) for char in word]
            words_indexs_list_lig.append(index)
        words_indexs_lig = np.array(words_indexs_list_lig)
        return words_indexs_lig
    def pro_index(seqs,
                  seqset=['G', 'A', 'V', 'L', 'I', 'P', 'F', 'Y', 'W', 'S', 'T', 'C', 'M', 'N', 'Q', 'K', 'R', 'H', 'D',
                          'E', '!']):
        char_to_int = dict((c, i) for i, c in enumerate(seqset))
        int_to_char = dict((i, c) for i, c in enumerate(seqset))
        seqs_new = [[word for word in re.findall(r'.{1}', str(document))] for document in seqs]
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
    def validation_fgfr4(model, target):
        base_file_path = r'/public/home/cadd1/liang/DTI/newmodel/validation/'
        data = pd.read_excel(base_file_path + target + '/' + target + '.xlsx', index_col=None, header=0)
        test_seq, test_smi = data.iloc[:, 1], data.iloc[:, 2]
        test_mol = [Chem.MolFromSmiles(smi) for smi in test_smi]
        y_test = data.iloc[:, 3]
        seqindex_test, smiindex_test = pro_index(test_seq), lig_index(test_smi)
        provec_test = np.array(w2v.feature_embeddings_protein(test_seq, 512).drop('index', axis=1))
        smivec_test = np.array(w2v.feature_embeddings_smiles(test_smi, 256).drop('index', axis=1))
        device = torch.device('cuda:3' if torch.cuda.is_available() else 'cpu')
        model.to(device)
        test_ligset = MolDataset(test_mol, y_test, 60)
        test_gcnloader = DataLoader(test_ligset)
        test_ligloader = DataLoader(smiindex_test)
        test_proloader = DataLoader(seqindex_test)
        test_smivecloader = DataLoader(smivec_test)
        test_provecloader = DataLoader(provec_test)
        model.eval()
        with torch.no_grad():
            model_pred = []
            for batch in zip(test_gcnloader, test_ligloader, test_proloader, test_smivecloader, test_provecloader):
                x, A, y, l, p, vecl, vecp = batch[0]['X'].float(), batch[0]['A'].float(), batch[0]['Y'].float(), \
                                            batch[1], batch[2], batch[3].float(), batch[4].float()
                x, A, y, l, p, vecl, vecp = x.to(device), A.to(device), y.to(device), l.to(device), p.to(
                    device), vecl.to(
                    device), vecp.to(device)
                pred_test = model(x, A, l, p, vecl, vecp).squeeze(-1)
                model_pred.append(pred_test.data.cpu().numpy())
        model_pred = np.concatenate(model_pred, -1)
        pred = model_pred
        r2 = r2_score(y_test, pred)
        r_test_set = pearsonr(y_test, pred)
        return r_test_set, y_test, pred
    r_fgfr4, true_fgfr4, pred_fgfr4 = validation_fgfr4(model, 'FGFR4')
    if abs(pred_fgfr4[8]-6.22) < 1.5 and abs(pred_fgfr4[9]-7.17) < 1.5:
        torch.save(model, 'model_' + str(epoch) + '.pkl')


    def validation_compare(model, target):
        base_file_path = r'/public/home/cadd1/liang/DTI/newmodel/validation/'
        data = pd.read_excel(base_file_path + target + '/' + target + '.xlsx', index_col=None, header=0)
        test_seq, test_smi = data.iloc[:, 1], data.iloc[:, 2]
        test_mol = [Chem.MolFromSmiles(smi) for smi in test_smi]
        y_test = data.iloc[:, 3]
        seqindex_test, smiindex_test = pro_index(test_seq), lig_index(test_smi)
        provec_test = np.array(w2v.feature_embeddings_protein(test_seq, 512).drop('index', axis=1))
        smivec_test = np.array(w2v.feature_embeddings_smiles(test_smi, 256).drop('index', axis=1))
        device = torch.device('cuda:3' if torch.cuda.is_available() else 'cpu')
        model.to(device)
        test_ligset = MolDataset(test_mol, y_test, 60)
        test_gcnloader = DataLoader(test_ligset)
        test_ligloader = DataLoader(smiindex_test)
        test_proloader = DataLoader(seqindex_test)
        test_smivecloader = DataLoader(smivec_test)
        test_provecloader = DataLoader(provec_test)
        model.eval()
        with torch.no_grad():
            model_pred = []
            for batch in zip(test_gcnloader, test_ligloader, test_proloader, test_smivecloader, test_provecloader):
                x, A, y, l, p, vecl, vecp = batch[0]['X'].float(), batch[0]['A'].float(), batch[0]['Y'].float(), \
                                            batch[1], batch[2], batch[3].float(), batch[4].float()

                x, A, y, l, p, vecl, vecp = x.to(device), A.to(device), y.to(device), l.to(device), p.to(
                    device), vecl.to(
                    device), vecp.to(device)
                pred_test = model(x, A, l, p, vecl, vecp).squeeze(-1)
                model_pred.append(pred_test.data.cpu().numpy())
        model_pred = np.concatenate(model_pred, -1)
        pred = model_pred
        r2 = r2_score(y_test, pred)
        r_test_set = pearsonr(y_test, pred)
        return r_test_set
    r_CDK2 = validation_compare(model, 'CDK2')
    r_MCL1 = validation_compare(model, 'MCL1')
    r_P381 = validation_compare(model, 'P381')
    r_BACE = validation_compare(model, 'BACE')
    r_Tyk2 = validation_compare(model, 'Tyk2')
    r_Thrombin = validation_compare(model, 'Thrombin')
    r_PTP1B = validation_compare(model, 'PTP1B')
    r_JNK1 = validation_compare(model, 'JNK1')
    if np.mean([r_CDK2,r_MCL1,r_P381,r_BACE,r_Tyk2,r_Thrombin,r_PTP1B,r_JNK1]) > 0.5:
        torch.save(model, 'model_' + str(epoch) + '.pkl')

    print(epoch,np.mean(np.array(train_loss)),val_loss)
    
end = time.time()
print('Time:',(end-start)/3600)

model.eval()
with torch.no_grad():
    true_train,true_test=[],[]
    pred_train,pred_test= [], []
    loss_train,loss_test = [],[]
    mae_train,mae_test = [],[]
    
    for batch in zip(train_gcnloader,train_ligloader,train_proloader,train_smivecloader,train_provecloader):
        x, A, y, l, p, vecl, vecp = batch[0]['X'].float(), batch[0]['A'].float(), batch[0]['Y'].float(), \
                         batch[1], batch[2], batch[3].float(), batch[4].float()
        x, A, y, l, p, vecl, vecp = x.to(device), A.to(device), y.to(device), l.to(device), p.to(device), vecl.to(device), vecp.to(device)
        pred = model(x,A,l,p,vecl,vecp).squeeze(-1)
        pred_train.append(pred.data.cpu().numpy())
        true_train.append(y.data.cpu().numpy())
        loss_train.append(loss_fn(y.float(),pred.float()).data.cpu().numpy())
        mae_train.append(np.mean(np.abs((y-pred).data.cpu().numpy())))
    
    for batch in zip(test_gcnloader,test_ligloader,test_proloader,test_smivecloader,test_provecloader):
        x, A, y, l, p, vecl, vecp = batch[0]['X'].float(), batch[0]['A'].float(), batch[0]['Y'].float(), \
                        batch[1], batch[2], batch[3].float(), batch[4].float()
        x, A, y, l, p, vecl, vecp = x.to(device), A.to(device), y.to(device), l.to(device), p.to(device), vecl.to(device), vecp.to(device)
        pred = model(x,A,l,p,vecl,vecp).squeeze(-1)
        pred_test.append(pred.data.cpu().numpy())
        true_test.append(y.data.cpu().numpy())
        loss_test.append(loss_fn(y.float(),pred.float()).data.cpu().numpy())
        mae_test.append(np.mean(np.abs((y-pred).data.cpu().numpy())))

pred_train = np.concatenate(pred_train,-1)
pred_test = np.concatenate(pred_test,-1)
true_train = np.concatenate(true_train,-1)
true_test = np.concatenate(true_test,-1)

print('batch_size:',batch_size)
print('lr:',lr)
print('Train loss:',np.mean(np.array(loss_train)), 'Test loss:',np.mean(np.array(loss_test)))
print('Train_r2:',r2_score(true_train,pred_train), 'Test_r2:',r2_score(true_test,pred_test))
print('Train_mae:',np.mean(np.array(mae_train)), 'Test_mae:',np.mean(np.array(mae_test)))
print('Train_rmse:',np.sqrt(np.mean(np.array(loss_train))), 'Test_rmse:',np.sqrt(np.mean(np.array(loss_test))))
print('Train_r:',pearsonr(true_train,pred_train))
print('Test_r:',pearsonr(true_test,pred_test))
torch.save(model,'model.pkl')

'''
import seaborn as sns
import matplotlib.pyplot as plt 
import numpy as np 
sns.set()

f = plt.figure(figsize=(7, 7))
gs = f.add_gridspec(2, 1)
x = np.linspace(0,200,199)
train_loss = []
test_loss = []
with open('model.txt') as f:
    li = f.readlines()
for i in range(1,200):
    loss = li[i+3].strip().split()
    train_loss.append(float(loss[1]))
    test_loss.append(float(loss[2]))

plt.plot(x,train_loss, label='Train Loss')
plt.plot(x,test_loss, label='Valid Loss')
#plt.title("Loss")
plt.legend()
'''
