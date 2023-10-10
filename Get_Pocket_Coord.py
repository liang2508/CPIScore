# -*- coding: utf-8 -*-

import math
import pandas as pd
import numpy as np
import time


def get_pro_atom_info(pdb_id,line):
    dic = {}
    atom=line.split()[-1].upper()
    dic['x'] = line[30:38]
    dic['y'] = line[38:46]
    dic['z'] = line[46:54]
    dic['atom'] = atom
    if len(line.split()[3]) == 3:
        dic['residue'] = line.split()[3]
        dic['atom'] = line.split()[2]
    elif len(line.split()[3]) == 1 or len(line.split()[2]) > 3:              #有的pdb氨基酸与原子编号相连
        dic['residue'] = line.split()[2][-3:]
        dic['atom'] = line.split()[2][0:2]       
    else:
        dic['residue'] = line.split()[3]
        dic['atom'] = line.split()[2]
    if line.split()[-7].isdigit():  
        dic['residueindex'] = line.split()[-7]
    elif len(line.split()[-7])>2 and line.split()[-7][0].isupper():         #有的pdb氨基酸编号中有字母（链）
        dic['residueindex'] = line.split()[-7]
    elif len(line.split()[-7])>2 and line.split()[-7][-1].isupper():
        dic['residueindex'] = line.split()[-7]
    else:
        dic['residueindex'] = line.split()[-6]
    return dic

def get_lig_atom_info(line):
    dic={}
    line=line.split()
    atom=line[5].upper()
    dic['x']=line[2]
    dic['y']=line[3]
    dic['z']=line[4]
    dic['atom']=atom
    return dic

def calc_distance(a,b):
    try:
        return math.sqrt((float(a[0])-float(b[0]))*(float(a[0])-float(b[0]))+
                         (float(a[1])-float(b[1]))*(float(a[1])-float(b[1]))+
                         (float(a[2])-float(b[2]))*(float(a[2])-float(b[2])))
    except:
        print(a,b)

def get_procoords(pdb_id,pro_file):
    protein = open(pro_file).readlines()
    coords = {}
    for pro_idx,pro_el in enumerate(protein):
        if not pro_el.startswith('ATOM'):
            continue
        if len(pro_el.split()[3]) == 1 and len(pro_el.split()[4]) == 1:
            continue
        if pro_el.split()[3] == 'UNK':
            continue
        pro_atom = get_pro_atom_info(pdb_id,pro_el)
        res = ['GLY','ALA','VAL','LEU','ILE','PRO','PHE','TRP','MET','TYR','SER','THR','CYS','ASN',
               'GLN','ASP','GLU','LYS','ARG','HIS']
        if pro_atom['residue'] not in res:
            continue
        k = pro_atom['residue'] + pro_atom['residueindex']
        if pro_atom['atom'] == 'CA':
            coords[k] = [pro_atom['x'],pro_atom['y'],pro_atom['z']]
        '''
        if k not in coords.keys():
            coords[k] = [[pro_atom['x'],pro_atom['y'],pro_atom['z']]]
        else:
            coords[k].append([pro_atom['x'],pro_atom['y'],pro_atom['z']])
    for i in coords.keys():
        li = coords[i]
        xli,yli,zli = [],[],[]
        for j in range(len(li)):
            xli.append(float(li[j][0]))
            yli.append(float(li[j][1]))
            zli.append(float(li[j][2]))
        coords[i] = []
        coords[i].append(np.mean(xli))
        coords[i].append(np.mean(yli))
        coords[i].append(np.mean(zli))
        '''
    return coords

def get_ligcoords(lig_name,lig_file):
    ligand = open(lig_file).readlines()
    coords = []
    meancoord = []
    for lig_idx, lig_el in enumerate(ligand):
        n = ligand.index('@<TRIPOS>ATOM\n') + 1
        if lig_idx < n:
            continue
        if lig_el.startswith('@<TRIPOS>BOND'):
            break
        if lig_el.split()[-4] == 'H':
            continue
        lig_atom = get_lig_atom_info(lig_el)  
        coords.append([lig_atom['x'],lig_atom['y'],lig_atom['z']])
    return coords
    '''
    xcoord = [float(i[0]) for i in coords]
    ycoord = [float(i[1]) for i in coords]
    zcoord = [float(i[2]) for i in coords]
    meancoord.append(np.mean(xcoord))
    meancoord.append(np.mean(ycoord))
    meancoord.append(np.mean(zcoord))
    distancemean = []
    for coord in coords:
        distancemean.append(calc_distance(coord,meancoord))
    cm = coords[distancemean.index(min(distancemean))]
    fm = coords[distancemean.index(max(distancemean))]
    distancefm = []
    for coord in coords:
        distancefm.append(calc_distance(coord,fm))
    ffm = coords[distancefm.index(max(distancefm))]
    return [meancoord,cm,fm,ffm]
    '''    
def calc_stat(data):
    meanvalue = np.mean(data)
    sigma = np.std(data)
    niu3 = 0.0
    for a in data:
        niu3 += a**3
    niu3 /= len(data) #这是E(X^3)    
    n = len(data)
    niu4 = 0.0
    for a in data:
        a -= meanvalue
        niu4 += a ** 4
    niu4 /= n
    skew = (niu3 - 3*meanvalue*sigma**2 - meanvalue**3)/(sigma**3)
    kurt =  niu4/(sigma**2)  #峰度
    return [meanvalue,sigma*sigma,skew] #返回了均值，方差，偏度

base_file_path = r'/public/home/cadd1/liang/DTI'
name = pd.read_excel(base_file_path+r'/total_data.xlsx')
Procoords = {}
Ligcoords = {}
df_li = pd.DataFrame(name.iloc[:,0]).values.tolist()
id_list = []
for code in df_li:
    id_list.append(code[0])
print('id number:',len(id_list))
i = 0
start = time.time()
for id in id_list:
    protein = base_file_path+'/total_complex_pocket_nowater/'+id+'.pdb'
    lig = base_file_path+'/total_ligands_mol2/'+id+'_ligand.mol2'
    Procoords[id] = get_procoords(id,protein)
    Ligcoords[id] = get_ligcoords(id,lig)


'''
df = pd.DataFrame()
prodes = {}
for id in Procoords.keys():
    rescoords = Procoords[id]
    meancoord = []
    allcoord = []
    desvec = []
    for res in rescoords.keys():
        allcoord.append(rescoords[res])
    xcoord = [i[0] for i in allcoord]
    ycoord = [i[1] for i in allcoord]
    zcoord = [i[2] for i in allcoord]
    meancoord.append(np.mean(xcoord))
    meancoord.append(np.mean(ycoord))
    meancoord.append(np.mean(zcoord))
    distancemean = []
    for coord in allcoord:
        distancemean.append(calc_distance(coord,meancoord))
    cm = allcoord[distancemean.index(min(distancemean))]
    fm = allcoord[distancemean.index(max(distancemean))]
    distancefm = []
    for coord in allcoord:
        distancefm.append(calc_distance(coord,fm))
    ffm = allcoord[distancefm.index(max(distancefm))]
    distancecm = []
    distanceffm = []
    for coord in allcoord:
        distancecm.append(calc_distance(coord,cm))
        distanceffm.append(calc_distance(coord,ffm))
    for v in calc_stat(distancemean):
        desvec.append(v)
    for v in calc_stat(distancecm):
        desvec.append(v)
    for v in calc_stat(distancefm):
        desvec.append(v)
    for v in calc_stat(distanceffm):
        desvec.append(v)
    prodes[id] = desvec
    df[id] = prodes[id]

df = df.T
df.to_excel(r'D:/ML/modelfinal/total_pocketusr.xlsx')
'''
