import math
import pdb
import numpy as np
import pandas as pd
# numpy打包工具
import pickle as pk
# 简化构建图结构的工具
import os, torch, sys
import Bio.PDB as PDB
from Bio.PDB.PDBParser import PDBParser
from biopandas.pdb import PandasPdb
from rdkit import Chem
import warnings

curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)
from data_process.protein_features import *

warnings.filterwarnings("ignore")

parser = PDBParser(PERMISSIVE=1)
# 定义氨基酸
Amino_acids = ['ALA', 'CYS', 'ASP', 'GLU', 'PHE', 'GLY', 'HIS', 'ILE', 'LYS', 'LEU', 'MET', 'ASN', 'PRO', 'GLN', 'ARG',
               'SER', 'THR', 'VAL', 'TRP', 'TYR']
# 定义缩写氨基酸
amino_acids = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y']
Amino_acids_num = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]
Amino_acids_dict = {}
for i in range(len(Amino_acids)):
    Amino_acids_dict[Amino_acids[i]] = Amino_acids_num[i]


def protein_to_int(protein):
    protein_int = []
    for i in range(len(protein)):
        temp = protein[i]
        index = [i for i, x in enumerate(amino_acids) if x == temp]
        if len(index) != 0:
            protein_int.append(index[0])
    return protein_int


def protein_to_onehot(seq):
    protein_to_int = dict((c, i) for i, c in enumerate(amino_acids))
    integer_encoded = [protein_to_int[char] for char in seq]
    # print(integer_encoded)
    onehot_encoded = []
    for value in integer_encoded:
        letter = [0 for _ in range(len(amino_acids))]
        letter[value] = 1
        onehot_encoded.append(letter)
    return onehot_encoded


def proteinOneHotProperties(pdbid):
    Amino_acid_list = []
    p_str = parser.get_structure(pdbid, 'data/protein_pdb/' + pdbid + '.pdb')
    for chain in p_str[0]:
        if chain.get_id() != ' ':
            for residue in chain:
                if residue.has_id('CA'):
                    if str(residue.get_resname()) in Amino_acids:
                        Amino_acid_list.append(residue.get_resname())
    protein_feature = []
    print(Amino_acid_list)
    for i in Amino_acid_list:
        num = Amino_acids_dict[i]
        feature = [0 for i in range(len(Amino_acids))]
        feature[num] = 1
        protein_feature.append(feature)
    return protein_feature


def get_path(pdbid, isTrain=True):
    path = ''
    pdbid = pdbid + '.pdb'
    if isTrain:
        if pdbid in os.listdir('data/train/ach/'):
            path = 'data/train/ach/'
        elif pdbid in os.listdir('data/train/ca/'):
            path = 'data/train/ca/'

        elif pdbid in os.listdir('data/train/k/'):
            path = 'data/train/k/'
        elif pdbid in os.listdir('data/train/na/'):
            path = 'data/train/na/'
    else:
        if pdbid in os.listdir('data/test/ach/'):
            path = 'data/test/ach/'
        elif pdbid in os.listdir('data/test/ca/'):
            path = 'data/test/ca/'

        elif pdbid in os.listdir('data/test/k/'):
            path = 'data/test/k/'
        elif pdbid in os.listdir('data/test/na/'):
            path = 'data/test/na/'
    return path


def getProteinGraph(pdbid, is_train=True):
    edge_list = []
    coordinate_list = []
    path = get_path(pdbid, is_train)
    pdb_pandas = PandasPdb().read_pdb(path + pdbid + '.pdb')
    atom_name_list = pdb_pandas.df['ATOM'].atom_name.tolist()
    atom_coordinate_list = np.array(pdb_pandas.df['ATOM'].loc[:, ('x_coord', 'y_coord', 'z_coord')]).tolist()
    residue_name_list = pdb_pandas.df['ATOM'].residue_name.tolist()
    protein_feature = []
    protein_sequence = ''
    num_residue = 0
    path3 = 'data/embedding.npz'
    datas = np.load(path3)
    count = 0
    for i in range(len(atom_name_list)):
        if atom_name_list[i] == 'CA':
            num_residue += 1
            protein_sequence += amino_acids[Amino_acids_dict[residue_name_list[i]]]
            coordinate_list.append(atom_coordinate_list[i])
    for i in protein_sequence:
        feature = [0 for i in range(len(Amino_acids))]
        feature[amino_acids.index(i)] = 1
        x = datas[pdbid]
        p = x[count]
        p =list(p)
        # p= [element for sublist in p for element in sublist]
        pho = VHSE[protein_letters_1to3[i]] + zScales[protein_letters_1to3[i]]
        feature = feature+pho+p
        count = count+1
        protein_feature.append(feature)
        #
        # if count == 10:
        #     print((np.array(feature)).shape)
        #     print(feature[0:30])


    for i in range(len(coordinate_list)):
        for j in range(len(coordinate_list)):
            dist = math.sqrt((atom_coordinate_list[i][0] - atom_coordinate_list[j][0]) ** 2 + (
                        atom_coordinate_list[i][1] - atom_coordinate_list[j][1]) ** 2 + (
                                         atom_coordinate_list[i][2] - atom_coordinate_list[j][2]) ** 2)
            if i != j and dist <= 7:
                edge_list.append([i, j])
    return num_residue, protein_feature, edge_list

# print(getProteinGraph('A0A3G3C7S9'))
