"""
Convert SMILES representation of molecules to Graphs
"""

import numpy as np
import sys
from rdkit import Chem


def adj_k(adj, k):
    """
    power of adjacent matrix
    :param adj:
    :param k:
    :return:
    """
    ret = adj
    for i in range(0, k-1):
        ret = np.dot(ret, adj)
    return convertAdj(ret)

def convertAdj(adj):
    """
    Change all non-zero elements in adj to 1.
    :param adj:
    :return:
    """
    dim = len(adj)
    a = adj.flatten()
    b = np.zeros(dim*dim)
    c = (np.ones(dim*dim)-np.equal(a,b)).astype('float64')
    d = c.reshape((dim, dim))
    return d


def convertToGraph(smi_lst, k):
    adj = []
    adj_norm = []
    features = []
    maxNumAtoms = 50 # Be careful here!
    for smi in smi_lst:
        iMol = Chem.MolFromSmiles(smi.rstrip())
        iAdjTmp = Chem.rdmolops.GetAdjacencyMatrix(iMol)
        # Feature
        if (iAdjTmp.shape[0] <= maxNumAtoms):
            iFeature = np.zeros((maxNumAtoms, 58))
            iFeatureTmp = []
            for atom in iMol.GetAtoms():
                iFeatureTmp.append(atom_feature(atom))
            iFeature[0:len(iFeatureTmp), 0:58] = iFeatureTmp # zerp pad for feature set
            features.append(iFeature)

            iAdj = np.zeros((maxNumAtoms, maxNumAtoms))
            iAdj[0:len(iFeatureTmp), 0:len(iFeatureTmp)] = iAdjTmp + np.eye(len(iFeatureTmp))
            adj.append(adj_k(np.asarray(iAdj), k))
    features = np.asarray(features)

    return adj, features


def atom_feature(atom):
    return np.array(one_of_k_encoding_unk(atom.GetSymbol(),
                                      ['C', 'N', 'O', 'S', 'F', 'H', 'Si', 'P', 'Cl', 'Br',
                                       'Li', 'Na', 'K', 'Mg', 'Ca', 'Fe', 'As', 'Al', 'I', 'B',
                                       'V', 'Tl', 'Sb', 'Sn', 'Ag', 'Pd', 'Co', 'Se', 'Ti', 'Zn',
                                       'Ge', 'Cu', 'Au', 'Ni', 'Cd', 'Mn', 'Cr', 'Pt', 'Hg', 'Pb']) +
                    one_of_k_encoding(atom.GetDegree(), [0, 1, 2, 3, 4, 5]) +
                    one_of_k_encoding_unk(atom.GetTotalNumHs(), [0, 1, 2, 3, 4]) +
                    one_of_k_encoding_unk(atom.GetImplicitValence(), [0, 1, 2, 3, 4, 5]) +
                    [atom.GetIsAromatic()])


def one_of_k_encoding(x, allowable_set):
    if x not in allowable_set:
        raise Exception("input {} not in allowable set {}.".format(x, allowable_set))
    return list(map(lambda s : x == s, allowable_set))


def one_of_k_encoding_unk(x, allowable_set):
    if x not in allowable_set:
        x = allowable_set[-1]
    return list(map(lambda s: x == s, allowable_set))


if __name__ == '__main__':
    dbName = 'CEP'
    length = 1024
    k = 1
    smiles_f = open('./'+dbName+'/smiles.txt')
    smiles_lst = smiles_f.readlines()
    print(len(smiles_lst))
    maxNum = int(len(smiles_lst)/length)
 #   adj, features = convertToGraph(smiles_lst, 1)
 #   print(np.asarray(features).shape)
 #   np.save('cep_adj.npy', adj)
 #   np.save('cep_feature.npy', features)

    for i in range(maxNum+1):
        lb = i*length
        ub = (i+1)*length
        adj, features = convertToGraph(smiles_lst[lb:ub], 1)
        print(np.asarray(features).shape)
        np.save('./'+dbName+'/adj/'+str(i)+'.npy', adj)
        np.save('./'+dbName+'/features/'+str(i)+'.npy', features)

