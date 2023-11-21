import math
import re
from Bio.PDB.PDBParser import PDBParser
import pickle
import networkx as nx
import numpy as np
protein_letters_1to3 = {
    "A": "Ala",
    "C": "Cys",
    "D": "Asp",
    "E": "Glu",
    "F": "Phe",
    "G": "Gly",
    "H": "His",
    "I": "Ile",
    "K": "Lys",
    "L": "Leu",
    "M": "Met",
    "N": "Asn",
    "P": "Pro",
    "Q": "Gln",
    "R": "Arg",
    "S": "Ser",
    "T": "Thr",
    "V": "Val",
    "W": "Trp",
    "Y": "Tyr",
}

identity = {
    'Ala': [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    'Arg': [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    'Asn': [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    'Asp': [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    'Cys': [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    'Gln': [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    'Glu': [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    'Gly': [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    'His': [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    'Ile': [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    'Leu': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    'Lys': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
    'Met': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
    'Phe': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
    'Pro': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
    'Ser': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
    'Thr': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
    'Trp': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
    'Tyr': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
    'Val': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]
}

VHSE = {
    'Ala': [0.15, -1.11, -1.35, -0.92, 0.02, -0.91, 0.36, -0.48],
    'Arg': [-1.47, 1.45, 1.24, 1.27, 1.55, 1.47, 1.30, 0.83],
    'Asn': [-0.99, 0.00, -0.37, 0.69, -0.55, 0.85, 0.73, -0.80],
    'Asp': [-1.15, 0.67, -0.41, -0.01, -2.68, 1.31, 0.03, 0.56],
    'Cys': [0.18, -1.67, -0.46, -0.21, 0.00, 1.20, -1.61, -0.19],
    'Gln': [-0.96, 0.12, 0.18, 0.16, 0.09, 0.42, -0.20, -0.41],
    'Glu': [-1.18, 0.40, 0.10, 0.36, -2.16, -0.17, 0.91, 0.02],
    'Gly':[-0.20,-1.53,-2.63,2.28,-0.53,-1.18,2.01,-1.34],
    'His': [-0.43, -0.25, 0.37, 0.19, 0.51, 1.28, 0.93, 0.65],
    'Ile': [1.27, -0.14, 0.30, -1.80, 0.30, -1.61, -0.16, -0.13],
    'Leu': [1.36, 0.07, 0.26, -0.80, 0.22, -1.37, 0.08, -0.62],
    'Lys': [-1.17, 0.70, 0.70, 0.80, 1.64, 0.67, 1.63, 0.13],
    'Met': [1.01, -0.53, 0.43, 0.00, 0.23, 0.10, -0.86, -0.68],
    'Phe': [1.52, 0.61, 0.96, -0.16, 0.25, 0.28, -1.33, -0.20],
    'Pro': [0.22, -0.17, -0.50, 0.05, -0.01, -1.34, -0.19, 3.56],
    'Ser': [-0.67, -0.86, -1.07, -0.41, -0.32, 0.27, -0.64, 0.11],
    'Thr': [-0.34, -0.51, -0.55, -1.06, -0.06, -0.01, -0.79, 0.39],
    'Trp': [1.50, 2.06, 1.79, 0.75, 0.75, -0.13, -1.01, -0.85],
    'Tyr': [0.61, 1.60, 1.17, 0.73, 0.53, 0.25, -0.96, -0.52],
    'Val': [0.76, -0.92, -0.17, -1.91, 0.22, -1.40, -0.24, -0.03]
}

zScales = {
    'Ala': [0.24, -2.32, 0.60, -0.14, 1.30],
    'Arg': [3.52, 2.50, -3.50, 1.99, -0.17],
    'Asn': [3.05, 1.62, 1.04, -1.15, 1.61],
    'Asp': [3.98, 0.93, 1.93, -2.46, 0.75],
    'Cys': [0.84, -1.67, 3.71, 0.18, -2.65],
    'Gln': [1.75, 0.50, -1.44, -1.34, 0.66],
    'Glu': [3.11, 0.26, -0.11, -3.04, -0.25],
    'Gly': [2.05, -4.06, 0.36, -0.82, -0.38],
    'His': [2.47, 1.95, 0.26, 3.90, 0.09],
    'Ile': [-3.89, -1.73, -1.71, -0.84, 0.26],
    'Leu': [-4.28, -1.30, -1.49, -0.72, 0.84],
    'Lys': [2.29, 0.89, -2.49, 1.49, 0.31],
    'Met': [-2.85, -0.22, 0.47, 1.94, -0.98],
    'Phe': [-4.22, 1.94, 1.06, 0.54, -0.62],
    'Pro': [-1.66, 0.27, 1.84, 0.70, 2.00],
    'Ser': [2.39, -1.07, 1.15, -1.39, 0.67],
    'Thr': [0.75, -2.18, -1.12, -1.46, -0.40],
    'Trp': [-4.36, 3.94, 0.59, 3.44, -1.59],
    'Tyr': [-2.54, 2.44, 0.43, 0.04, -1.47],
    'Val': [-2.59, -2.64, -1.54, -0.85, -0.02]
}
# label = np.array([1, 0, 0, 0, 0, 0],
#                  [0, 1, 0, 0, 0, 0],
#                  [0, 0, 1, 0, 0, 0],
#                  [0, 0, 0, 1, 0, 0],
#                  [0, 0, 0, 0, 1, 0],
#                  [0, 0, 0, 0, 0, 1])


def distance(coord1, coord2):
    x1, y1, z1 = coord1
    x2, y2, z2 = coord2
    val = (x1 - x2) ** 2 + (y1 - y2) ** 2 + (z1 - z2) ** 2
    dis = round(math.sqrt(val), 2)
    return dis


def load_pdb(filename):
    p = PDBParser(PERMISSIVE=1)
    pdb_id = filename[0:-4]
    print(pdb_id)
    pdb = p.get_structure(pdb_id, filename)
    residues = pdb.get_residues()
    res_dic = {}
    for residue in residues:
        seq_id = re.findall('resseq=(\d*)', str(residue.__repr__))[0]
        for atom in residue.get_atoms():
            if atom.get_fullname().strip() == 'CA':
                coords = str(atom.get_coord())[1:-1].split()
                coords = [i.strip() for i in coords]
                coords = [float(i) for i in coords]
                res_dic[seq_id] = (residue.get_resname().lower().capitalize(), coords)
    # print('loading')
    return res_dic


def res2node(res_dic):
    # print('in res2node')
    nodes = []
    for k, v in res_dic.items():
        node = int(k)
        feature = []
        # 20d identity one-hot
        feature.extend(identity[v[0]])
        # 8d VHSE one-hot
        feature.extend(VHSE[v[0]])
        # 5d zScales one-hot
        feature.extend(zScales[v[0]])
        # print('-------------------------------------------')
        # print(nodes)
        # print('--------------------------------------------')
        nodes.append((node, {'feature': feature}))
    # print('res2node is running')
    return nodes


def res2edge(res_dic):
    # print("in res2edge")
    edges = []
    for k, v in res_dic.items():
        node1 = int(k)
        for key, val in res_dic.items():
            node2 = int(key)
            if node1 == node2:
                continue
            else:
                coord1 = val[1]
                coord2 = v[1]
                dis = distance(coord1, coord2)
                # threshold
                if dis < 7:
                    edges.append((node1, node2))
    # print('res2edge is running')
    return edges


def build_graph(nodes, edges):
    # print('loading build_graph:')
    g = nx.Graph()
    g.add_nodes_from(nodes)
    g.add_edges_from(edges)
    # print(g)
    return g

def save_pkl(graph, filepath):
    # print('saving')
    # print(graph)
    pickle.dump(graph, open(filepath, 'wb'))


def read_pkl(filepath):
    print('reading')
    return pickle.load(open(filepath, 'rb'))
