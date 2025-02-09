import glob
import math
import numpy as np
import dill as pkl
from collections import defaultdict

import torch
from torch.nn import functional as F
from torch.utils.data import Dataset, Subset, DataLoader
import pytorch_lightning as pl

class GraphDataset(Dataset):

    def __init__(self, fname, n_graph, n_isomorph, n_node, include_graphs=None):

        '''
        args
        ----
        fname : str
            name of the dataset file
        n_graph : int
            number of graphs to include from the dataset
        n_isomorph : int
            number of isomorphs to include from each graph
        n_node : int
            (max) number of nodes in each graph (to make pad token integers)
        include_graphs : list of int
            list of graph numbers to include in the dataset
        '''

        # sample a subset of graphs
        self.n_graph = n_graph
        self.n_isomorph = n_isomorph
        self.include_graphs = include_graphs
        self.setup_raw_data(fname)

        self.max_graph_edges = max([len(edges) for edges in self.raw_data['edge_list']])
        self.max_path_len = max([ # indicates max number of nodes on path
            len(shortest_paths[0])
            for shortest_paths in self.raw_data['shortest_path']
        ])
        self.input_vocab = list(range(n_node)) + ['pad']
        self.pad_index = self.input_vocab.index('pad')
    
    def setup_raw_data(self, fname):
        self.raw_data = pkl.load(open(fname, 'rb'))
        # use self.n_graph and self.n_isomorph to sample a subset of graphs to learn
        if self.include_graphs is None:
            self.include_graphs = np.random.choice(
                np.unique(self.raw_data['graph_num']),
                size=self.n_graph,
                replace=False
            ).tolist()
            self.include_graphs.sort()
        indices = np.in1d(self.raw_data['graph_num'], self.include_graphs)
        indices &= np.in1d(self.raw_data['isomorph_num'], list(range(self.n_isomorph)))
        for key in self.raw_data.keys():
            self.raw_data[key] = np.array(self.raw_data[key])[indices].tolist()

    def __len__(self):
        return len(self.raw_data['start_goal'])

    def __getitem__(self, index):

        # read from raw dataset, turn into tensors
        # and pad varaible length vectors (e.g, graph edge tokens, shortest path answers)

        graph_num = torch.tensor(self.raw_data['graph_num'][index])
        isomorph_num = torch.tensor(self.raw_data['isomorph_num'][index])
        start_goal_probe = torch.tensor(self.raw_data['start_goal'][index])
        path_len = torch.tensor(self.raw_data['path_length'][index])
        
        edges = torch.tensor(self.raw_data['edge_list'][index]) # (n, 2)
        edges = F.pad(edges, 
                    pad=(0, 0, 0, self.max_graph_edges - edges.shape[0]), 
                    value=self.pad_index)

        # for problems with multiple shortest paths, we take a random shortest path
        raw_shortest_paths = self.raw_data['shortest_path'][index]
        path = raw_shortest_paths[np.random.choice(len(raw_shortest_paths))]
        shortest_path = torch.tensor(path)
        shortest_path = F.pad(shortest_path, 
                              pad=(0, self.max_path_len - shortest_path.shape[0]),
                              value=self.pad_index)

        return {
            'index': index,
            'graph_num': graph_num,
            'isomorph_num': isomorph_num,
            'graph_context': self.include_graphs.index(graph_num) * self.n_isomorph + isomorph_num, # (1)
            'edges': edges, # (max_graph_edges, 2)
            'start_goal_probe': start_goal_probe, # (2)
            'path_len': path_len,
            'shortest_path': shortest_path, # (max_path_len)
        }

class GraphDataModule(pl.LightningDataModule):

    def __init__(self, dataset, batch_size, split_spec, train_idx=None, val_idx=None):

        super().__init__()

        assert type(dataset) == GraphDataset
        assert batch_size < len(dataset)

        self.dataset = dataset
        self.batch_size = batch_size
        self.split_spec = split_spec
        self.train_idx = train_idx
        self.val_idx = val_idx

        self.setup()

    def setup(self, stage=None):

        if stage == 'fit' or stage is None:
            if self.train_idx is None and self.val_idx is None:
                train_idx, val_idx = self.split_dataset()
                self.train_idx = train_idx
                self.val_idx = val_idx
            self.data_train = Subset(self.dataset, self.train_idx)
            self.data_val = Subset(self.dataset, self.val_idx)

        if stage == 'test' or stage is None:
            self.data_test = self.dataset

    def split_dataset(self):

        '''
        generates train_idx and val_idx based on self.split_spec
        returns
        -------
        train_idx : list of int
            indicates the indices of observations used for training
        val_idx : list of int
            indicates the indices of observations used in validation
        '''

        N_seq = len(self.dataset)
        all_seq_idx = np.arange(N_seq)

        # randomly hold out some sequences
        if self.split_spec['mode'] == 'random':
            n_train = math.floor(N_seq * self.split_spec['train_prop'])
            train_idx = np.random.choice(all_seq_idx, size=n_train, replace=False)
        
        elif self.split_spec['mode'] == 'by_isomorph':
            # split problems by isomorphic graphs (random isomorphs from each graph)
            isomorph_num = [self.dataset.raw_data['isomorph_num'][i] for i in all_seq_idx]
            train_graphs = np.random.choice(np.unique(isomorph_num), size=int(len(np.unique(isomorph_num))*self.split_spec['train_prop']), replace=False)
            train_idx = all_seq_idx[np.in1d(isomorph_num, train_graphs)]
        
        elif self.split_spec['mode'] == 'by_graph':
            # split problems by graph (always hold out same graphs across runs)
            graph_num = [self.dataset.raw_data['graph_num'][i] for i in all_seq_idx]
            train_graphs = np.unique(graph_num)[:int(len(np.unique(graph_num))*self.split_spec['train_prop'])]
            # alternative: hold out random set of graphs for different runs, but performance can vary more due to different held-out graph difficulties
            # train_graphs = np.random.choice(np.unique(graph_num), size=int(len(np.unique(graph_num))*self.split_spec['train_prop']), replace=False)
            train_idx = all_seq_idx[np.in1d(graph_num, train_graphs)]

        else:
            raise ValueError('unsupported split mode')

        val_idx = all_seq_idx[~np.isin(all_seq_idx, train_idx)]
        # turn into lists for wandb to record all values
        # the sort is for the human eye...
        return np.sort(train_idx).tolist(), np.sort(val_idx).tolist()
    
    def train_dataloader(self):
        return DataLoader(self.data_train, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.data_val, batch_size=self.batch_size, num_workers=4)

    def test_dataloader(self):
        return DataLoader(self.data_test, batch_size=self.batch_size, num_workers=4)