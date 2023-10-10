#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from sklearn.metrics import mean_squared_error,r2_score,mean_absolute_error
from scipy.stats import pearsonr
import time
import torch 
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.optim.lr_scheduler as lr_scheduler
import os
os.chdir('F:\\DTI\\CPIScore\\model')
import sys
sys.path.append('F:\\DTI\\CPIScore\\model')
from model import Model
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

lig_train = calc_des.smiindex_train
lig_test = calc_des.smiindex_test
prot_train = calc_des.seqindex_train
prot_test = calc_des.seqindex_test

y_train = calc_des.train_prop
y_test = calc_des.test_prop

model = Model()
#device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

batch_size = 32
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
    if epoch == 100 or epoch == 150 or epoch == 180:
        torch.save(model, './result/model_' + str(epoch) + '.pkl')
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
    torch.save(model,'./result/model_'+str(epoch)+'.pkl')
    
end = time.time()
print('Time:',(end-start)/3600)

model.eval()
with torch.no_grad():
    true_train,true_test=[],[]
    pred_train,pred_test= [], []
    
    for batch in zip(train_gcnloader,train_ligloader,train_proloader,train_smivecloader,train_provecloader):
        x, A, y, l, p, vecl, vecp = batch[0]['X'].float(), batch[0]['A'].float(), batch[0]['Y'].float(), \
                         batch[1], batch[2], batch[3].float(), batch[4].float()
        x, A, y, l, p, vecl, vecp = x.to(device), A.to(device), y.to(device), l.to(device), p.to(device), vecl.to(device), vecp.to(device)
        pred = model(x,A,l,p,vecl,vecp).squeeze(-1)
        pred_train.append(pred.data.cpu().numpy())
        true_train.append(y.data.cpu().numpy())
    
    for batch in zip(test_gcnloader,test_ligloader,test_proloader,test_smivecloader,test_provecloader):
        x, A, y, l, p, vecl, vecp = batch[0]['X'].float(), batch[0]['A'].float(), batch[0]['Y'].float(), \
                        batch[1], batch[2], batch[3].float(), batch[4].float()
        x, A, y, l, p, vecl, vecp = x.to(device), A.to(device), y.to(device), l.to(device), p.to(device), vecl.to(device), vecp.to(device)
        pred = model(x,A,l,p,vecl,vecp).squeeze(-1)
        pred_test.append(pred.data.cpu().numpy())
        true_test.append(y.data.cpu().numpy())

pred_train, true_train = np.concatenate(pred_train,-1), np.concatenate(true_train,-1)
pred_test, true_test = np.concatenate(pred_test,-1), np.concatenate(true_test,-1)
mae_train = mean_absolute_error(true_train,pred_train)   # np.mean(np.abs((true_train-pred_train)))
mae_test = mean_absolute_error(true_test,pred_test)      # np.mean(np.abs((true_test-pred_test)))
rmse_train = (mean_squared_error(true_train,pred_train))**0.5
rmse_test = (mean_squared_error(true_test,pred_test))**0.5

print('batch_size:',batch_size)
print('lr:',lr)
print('Train_r2:',r2_score(true_train,pred_train), 'Test_r2:',r2_score(true_test,pred_test))
print('Train_mae:',mae_train, 'Test_mae:',mae_test)
print('Train_rmse:',rmse_train, 'Test_rmse:',rmse_test)
print('Train_r:',pearsonr(true_train,pred_train))
print('Test_r:',pearsonr(true_test,pred_test))


import seaborn as sns
import matplotlib.pyplot as plt 
import numpy as np 
sns.set()

f = plt.figure(figsize=(7, 7))
gs = f.add_gridspec(2, 1)
x = np.linspace(1,100,100)
plt.plot(x,train_loss_epoch[:100], label='Train Loss')
plt.plot(x,val_loss_epoch[:100], label='Valid Loss')
#plt.title("Loss")
plt.legend()
plt.savefig('./result/loss_100.png',bbox_inches='tight')
plt.show()

