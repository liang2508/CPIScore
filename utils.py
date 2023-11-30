#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 21 14:40:32 2021

@author: dycomp
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from rdkit import Chem
from rdkit.Chem import MACCSkeys
from rdkit.Chem.rdmolops import GetAdjacencyMatrix
import scipy.sparse as sp

np.random.seed(1995)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, device, stride=1):
        super(ResidualBlock, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.conv1 = nn.Conv1d(self.in_channels, self.out_channels, kernel_size=5,
                     stride=stride, padding=2, bias=False)
        self.bn1 = nn.BatchNorm1d(self.out_channels)
        self.elu = nn.ELU(inplace=True)
        self.conv2 = nn.Conv1d(self.out_channels, self.in_channels, kernel_size=3,
                     stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(self.in_channels)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.elu(self.bn1(out))
        out = self.conv2(out)
        out = self.bn2(out)
        out += residual
        out = self.elu(out)      
        return out

class SelfAttention(nn.Module):
    def __init__(self, embed_size, heads):
        super(SelfAttention, self).__init__()
        self.embed_size = embed_size
     #   self.embed_size = int(forward_expansion*embed_size)
        self.heads = heads
        self.head_dim = self.embed_size // heads
        
        assert(self.head_dim * heads == self.embed_size), "Embed size needs to be div by heads"
        
        self.values = nn.Linear(self.head_dim, self.head_dim, bias = False)
        self.keys = nn.Linear(self.head_dim, self.head_dim, bias = False)
        self.queries = nn.Linear(self.head_dim, self.head_dim, bias = False)
        self.fc_out = nn.Linear(heads*self.head_dim, self.embed_size)
        
    def forward(self, values, keys, query,mask):
        N = query.shape[0]
        value_len, key_len, query_len = values.shape[1], keys.shape[1], query.shape[1]
        values = values.reshape(N, value_len, self.heads, self.head_dim)
        keys = keys.reshape(N, key_len, self.heads, self.head_dim)
        queries = query.reshape(N, query_len, self.heads, self.head_dim)
        values = self.values(values)
        keys = self.keys(keys)
        queries = self.queries(queries)
        energy = torch.einsum('nqhd,nkhd -> nhqk', [queries, keys])
        
        if mask is not None:
            energy = energy.masked_fill(mask == 0, float('-1e10'))
            
        attention = torch.softmax(energy/(self.embed_size**(1/2)),dim=2)
        
        out = torch.einsum('nhql,nlhd -> nqhd',[attention, values]).reshape(
                N,query_len,self.heads*self.head_dim)
        #attention shape:(N,heads,query_len,key_len) == energy shape
        #values shape:(N,value_len,heads,heads_dim)
        #after einsum:(N,query_len,heads,head_dim) then flatten last two dimensions
        
        out = self.fc_out(out)
        return out

class TransformerBlock(nn.Module):
    def __init__(self,embed_size,heads,dropout,forward_expansion,device):
        super(TransformerBlock,self).__init__()
        self.attention = SelfAttention(embed_size,heads)
        self.n = nn.Linear(embed_size,int(0.5*embed_size))
        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)
      #  self.norm2 = nn.LayerNorm(int(0.5*embed_size))
        '''
        self.feed_forward = nn.Sequential(
                nn.Linear(embed_size,forward_expansion*embed_size),
                nn.ReLU(),
                nn.Linear(forward_expansion*embed_size,embed_size)
                )
        '''
        '''
        self.feed_forward256 = nn.Sequential(
                nn.Conv1d(256,int(forward_expansion*256),kernel_size=5,padding=2),
                nn.BatchNorm1d(int(forward_expansion*256)),
                nn.ReLU(),
                nn.Conv1d(int(forward_expansion*256),256,kernel_size=3,padding=1),
                nn.BatchNorm1d(256),
                nn.Conv1d(256,256,kernel_size=2,stride=2),   #downsample
                nn.BatchNorm1d(256)
                )  
        self.feed_forward512 = nn.Sequential(
                nn.Conv1d(512,int(forward_expansion*512),kernel_size=5,padding=2),
                nn.BatchNorm1d(int(forward_expansion*512)),
                nn.ReLU(),
                nn.Conv1d(int(forward_expansion*512),512,kernel_size=3,padding=1),
                nn.BatchNorm1d(512),
                nn.Conv1d(512,512,kernel_size=2,stride=2),   #downsample
                nn.BatchNorm1d(512)
                )     
        '''
        self.layer256 = self.make_layer(ResidualBlock, 256, 128, 4)
        self.layer512 = self.make_layer(ResidualBlock, 512, 256, 4)
        self.dropout = nn.Dropout(dropout)
        
    def make_layer(self, block, in_channels, out_channels, cnn_layers, stride=1):
        layers = []
        for i in range(0, cnn_layers):
            layers.append(block(in_channels, out_channels, device=device))
        return nn.Sequential(*layers)
        
    def forward(self,value,key,query,mask):
        attention = self.attention(value,key,query,mask)
        x = self.dropout(self.norm1(attention + query))
        '''
        if x.shape[1] == 256:
            forward = self.feed_forward256(x)
        else:
            forward = self.feed_forward512(x)
        x = self.n(x)
        '''
        if x.shape[1] == 256:
            forward = self.layer256(x)
            forward = self.layer256(forward)
       #     forward = self.down256(forward)
        else:
            forward = self.layer512(x)
            forward = self.layer512(forward)     
        #    forward = self.down512(forward)
    #    x = self.n(x)
        out = self.dropout(self.norm2(forward+x))
        return out

class Encoder(nn.Module):
    def __init__(
            self,
            embed_size,
            num_layers,
            heads,
            forward_expansion,
            dropout,
            device):
        super(Encoder,self).__init__()
        self.embed_size = embed_size
        self.device = device
        self.word_embedding = nn.Embedding(21, embed_size)
        self.position_embedding = nn.Embedding(512, embed_size)
        self.layers = nn.ModuleList(
            [
                TransformerBlock(
                     embed_size,
                     heads,
                     dropout=dropout,
                     forward_expansion=forward_expansion,
                     device = device
                ) 
                for _ in range(num_layers)
            ]
        )
        self.dropout = nn.Dropout(dropout)
    def forward(self,x,mask):
        N, seq_length = x.shape
        positions = torch.arange(0, seq_length).expand(N, seq_length).to(self.device)
        out = self.dropout(self.word_embedding(x) + self.position_embedding(positions))
        for layer in self.layers:
            out = layer(out, out, out,mask)
        return out

class DecoderBlock(nn.Module):
    def __init__(self, embed_size, heads, forward_expansion, dropout, device):
        super(DecoderBlock, self).__init__()
        self.attention = SelfAttention(embed_size, heads)
        self.n = nn.Linear(int(0.5*embed_size),embed_size)
        self.norm = nn.LayerNorm(embed_size)
        self.transformer_block = TransformerBlock(
             embed_size, heads, dropout, forward_expansion, device      
        )
        self.dropout = nn.Dropout(dropout)
        self.embed_size = embed_size
    
    def forward(self, x, value, key, pro_mask, lig_mask):
     #   print('xshape:',x.shape)
        attention = self.attention(x, x, x, lig_mask)
        query = self.dropout(self.norm(attention + x))
        if value.shape[-1] != self.embed_size:
            value = self.n(value)
        if key.shape[-1] != self.embed_size:
            key = self.n(key)
        out = self.transformer_block(value, key, query, pro_mask)
        return out

class Decoder(nn.Module):
    def __init__(self,
                 embed_size,
                 num_layers,
                 heads,
                 device,
                 forward_expansion,
                 dropout):
        super(Decoder, self).__init__()
        self.embed_size = embed_size
        self.device = device
        self.n = nn.Linear(int(0.5*self.embed_size),self.embed_size)
        self.word_embedding = nn.Embedding(38,embed_size)
        self.position_embedding = nn.Embedding(256, embed_size)
        self.layers = nn.ModuleList(
                [DecoderBlock(embed_size, heads, forward_expansion, dropout, device)
                        for _ in range(num_layers)]
        )
        self.fc_out1 = nn.Linear(256,512)
        self.fc_out2 = nn.Linear(512,256)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, enc_out, pro_mask, lig_mask):
        N, seq_length = x.shape
        positions = torch.arange(0, seq_length).expand(N, seq_length).to(self.device)
        x = self.dropout(self.word_embedding(x) + self.position_embedding(positions))
    #    print('xshapea:',x.shape)
        for layer in self.layers:
            if x.shape[2] != self.embed_size:
                x = self.n(x)
            x = layer(x, enc_out, enc_out, pro_mask, lig_mask)
        x = x.mean(1)
        out = self.fc_out1(x)
        out = self.fc_out2(out)
        return out

class Transformer(nn.Module):
    def __init__(
            self,
            device,
            embed_size,
            num_layers,
            heads,
            forward_expansion,
            dropout=0.5):
        super(Transformer, self).__init__()
        self.decoder = Decoder(
                embed_size,
                num_layers,
                heads,
                device,
                forward_expansion,
                dropout)
        self.encoder = Encoder(
                embed_size,
                num_layers,
                heads,
                forward_expansion,
                dropout,
                device)
        self.pro_pad_idx = 20
        self.lig_pad_idx = 37
        self.device = device

    def make_pro_mask(self, pro):
        pro_mask = (pro != self.pro_pad_idx).unsqueeze(1).unsqueeze(2)
        #(N, 1, 1, pro_len)
        return pro_mask.to(self.device)
    
    def make_lig_mask(self, lig):
        N, lig_len = lig.shape
        lig_mask = torch.tril(torch.ones((lig_len, lig_len))).expand(
                N, 1, lig_len, lig_len)
        return lig_mask.to(self.device)

    def forward(self, pro, lig):
        pro_mask = self.make_pro_mask(pro)
        lig_mask = self.make_lig_mask(lig)
        enc_pro = self.encoder(pro, pro_mask)
    #    print('lig_shape:',lig.shape,'encpro_shape:',enc_pro.shape)
        out = self.decoder(lig, enc_pro, pro_mask, lig_mask)
        return out


class MolDataset(Dataset):
    def __init__(self, mol, label, max_natoms):
        self.mol = mol
        self.prop = label
        self.max_natoms = max_natoms

    def __len__(self):
        return len(self.mol)

    def __getitem__(self, idx):
        m = self.mol[idx]
        natoms = m.GetNumAtoms()

        fp = self.finger(m)
        A = GetAdjacencyMatrix(m) + np.eye(natoms)
        Norm_A = self.normalize_adj(A)
        A_padding = np.zeros((self.max_natoms, self.max_natoms))
        A_padding[:natoms, :natoms] = Norm_A

        X = [self.atom_feature(m, i) for i in range(natoms)]
        for i in range(natoms, self.max_natoms):
            X.append(np.zeros(37))
        X = np.array(X)

        sample = dict()
        sample['X'] = torch.from_numpy(X)
        sample['A'] = torch.from_numpy(A_padding)
        sample['f'] = torch.from_numpy(fp)
        sample['Y'] = self.prop[idx]
        return sample

    def finger(self, mol):
        fp = MACCSkeys.GenMACCSKeys(mol)
        #  fp = AllChem.GetMorganFingerprintAsBitVect(mol,2,nBits=1024)
        fp = fp.ToBitString()
        fp = np.array(list(fp)).astype('int8')
        return fp

    def normalize_adj(self, adj):
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

    def one_hot(self, x, allowable_set):
        if x not in allowable_set:
            raise Exception('input {0} not in allowable set{1}'.format(x, allowable_set))
        return list(map(lambda s: x == s, allowable_set))

    def one_hot_pad(self, x, allowable_set):
        if x not in allowable_set:
            x = allowable_set[-1]
        return list(map(lambda s: x == s, allowable_set))

    def NullToZero(self, x):
        if np.isnan(x) or np.isinf(x):
            x = 0
        else:
            x = x
        return x

    def atom_feature(self, m, atom_i):
        hyb = list(Chem.rdchem.HybridizationType.names.values())
        atom = m.GetAtomWithIdx(atom_i)
        Chem.rdPartialCharges.ComputeGasteigerCharges(m)
        return np.array(self.one_hot_pad(atom.GetSymbol(), ['C', 'N', 'O', 'S', 'F', 'Cl', 'Br', 'I', 'H']) +
                        self.one_hot(atom.GetDegree(), [0, 1, 2, 3, 4, 5]) +
                        self.one_hot(atom.GetTotalNumHs(), [0, 1, 2, 3, 4]) +
                        self.one_hot(atom.GetImplicitValence(), [0, 1, 2, 3, 4, 5]) +
                        self.one_hot(atom.GetHybridization(), hyb) +
                        [atom.GetIsAromatic()] + [atom.IsInRing()] + [
                            self.NullToZero(atom.GetDoubleProp('_GasteigerCharge'))])
