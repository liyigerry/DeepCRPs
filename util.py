import h5py, math, os, torch
import pandas as pd
import numpy as np

from Bio import SeqIO
from torch.utils.data import Dataset
from torch_geometric.data import InMemoryDataset
from torch_geometric.loader import DataLoader
from torch_geometric import data as DATA
from torch_geometric.loader import DataLoader



class ProteinDataset(InMemoryDataset):
    def __init__(self, root='/tmp', dataset=None, protein=None, affinity=None, transform=None, pre_transform=None, protein_graph=None):
        super(ProteinDataset, self).__init__(root, transform, pre_transform)
        self.dataset = dataset

        if os.path.isfile(self.processed_paths[0]):
            print('Pre-processd data found: {}, loading ...'.format(self.processed_paths[0]))
            self.data, self.slices = torch.load(self.processed_paths[0])
        else:
            print('Pre-processd data not found: {}, doing pre-processing ...'.format(self.processed_paths[0]))
            self.process(protein, affinity, protein_graph)
            self.data, self.slices = torch.load(self.processed_paths[0])
    
    @property
    def raw_file_names(self):
        pass
    @property
    def processed_file_names(self):
        return [self.dataset + '.pt']
    
    def download(self):
        # download_url(url='', self.raw_dir)
        pass
    def _download(self):
        pass

    def _process(self):
        if not os.path.exists(self.processed_dir):
            os.makedirs(self.processed_dir)
    
    def process(self, protein, affinity, protein_graph):
        assert (len(protein) ==  len(affinity)), '这两个列表必须是相同的长度!'
        data_list = []
        data_len = len(protein)
        for i in range(data_len):
            print('将分子格式转换为图结构：{}/{}'.format(i + 1, data_len))
            target = protein[i]
            label = float(affinity[i])
            print(target)
            print(label)

            size, features, edge_index = protein_graph[i][target]
            GCNProtein = DATA.Data(x=torch.Tensor(features), edge_index=torch.LongTensor(edge_index).transpose(-1, 0), y=torch.IntTensor([label]), id=target)
            GCNProtein.__setitem__('size', torch.LongTensor([size]))
            data_list.append(GCNProtein)

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]
        
        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        print('将构建完的图信息保存到文件中')
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])
