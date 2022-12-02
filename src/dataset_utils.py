#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8

import torch

import os
import os.path as osp
import pickle
import torch_geometric.transforms as T

import numpy as np

from cSBM_dataset import dataset_ContextualSBM
from torch_geometric.datasets import Planetoid
from torch_geometric.datasets import Amazon
from torch_geometric.datasets import WikipediaNetwork, WebKB
from torch_geometric.datasets import Actor
from torch_sparse import coalesce
from torch_geometric.data import InMemoryDataset, download_url, Data
from torch_geometric.utils.undirected import to_undirected


# class dataset_heterophily(InMemoryDataset):
#     def __init__(self, root='data/', name=None,
#                  p2raw=None,
#                  train_percent=0.01,
#                  transform=None, pre_transform=None):

#         existing_dataset = ['chameleon', 'film', 'squirrel']
#         if name not in existing_dataset:
#             raise ValueError(
#                 f'name of hypergraph dataset must be one of: {existing_dataset}')
#         else:
#             self.name = name

#         self._train_percent = train_percent

#         if (p2raw is not None) and osp.isdir(p2raw):
#             self.p2raw = p2raw
#         elif p2raw is None:
#             self.p2raw = None
#         elif not osp.isdir(p2raw):
#             raise ValueError(
#                 f'path to raw hypergraph dataset "{p2raw}" does not exist!')

#         if not osp.isdir(root):
#             os.makedirs(root)

#         self.root = root

#         super(dataset_heterophily, self).__init__(
#             root, transform, pre_transform)

#         self.data, self.slices = torch.load(self.processed_paths[0])
#         self.train_percent = self.data.train_percent

#     @property
#     def raw_dir(self):
#         return osp.join(self.root, self.name, 'raw')

#     @property
#     def processed_dir(self):
#         return osp.join(self.root, self.name, 'processed')

#     @property
#     def raw_file_names(self):
#         file_names = [self.name]
#         return file_names

#     @property
#     def processed_file_names(self):
#         return ['data.pt']

#     def download(self):
#         pass

#     def process(self):
#         p2f = osp.join(self.raw_dir, self.name)
#         with open(p2f, 'rb') as f:
#             data = pickle.load(f)
#         data = data if self.pre_transform is None else self.pre_transform(data)
#         torch.save(self.collate([data]), self.processed_paths[0])

#     def __repr__(self):
#         return '{}()'.format(self.name)


# class WebKB(InMemoryDataset):
#     r"""The WebKB datasets used in the
#     `"Geom-GCN: Geometric Graph Convolutional Networks"
#     <https://openreview.net/forum?id=S1e2agrFvS>`_ paper.
#     Nodes represent web pages and edges represent hyperlinks between them.
#     Node features are the bag-of-words representation of web pages.
#     The task is to classify the nodes into one of the five categories, student,
#     project, course, staff, and faculty.

#     Args:
#         root (string): Root directory where the dataset should be saved.
#         name (string): The name of the dataset (:obj:`"Cornell"`,
#             :obj:`"Texas"` :obj:`"Washington"`, :obj:`"Wisconsin"`).
#         transform (callable, optional): A function/transform that takes in an
#             :obj:`torch_geometric.data.Data` object and returns a transformed
#             version. The data object will be transformed before every access.
#             (default: :obj:`None`)
#         pre_transform (callable, optional): A function/transform that takes in
#             an :obj:`torch_geometric.data.Data` object and returns a
#             transformed version. The data object will be transformed before
#             being saved to disk. (default: :obj:`None`)
#     """

#     url = ('https://raw.githubusercontent.com/graphdml-uiuc-jlu/geom-gcn/'
#            'master/new_data')

#     def __init__(self, root, name, transform=None, pre_transform=None):
#         self.name = name.lower()
#         assert self.name in ['cornell', 'texas', 'washington', 'wisconsin']

#         super(WebKB, self).__init__(root, transform, pre_transform)
#         self.data, self.slices = torch.load(self.processed_paths[0])

#     @property
#     def raw_dir(self):
#         return osp.join(self.root, self.name, 'raw')

#     @property
#     def processed_dir(self):
#         return osp.join(self.root, self.name, 'processed')

#     @property
#     def raw_file_names(self):
#         return ['out1_node_feature_label.txt', 'out1_graph_edges.txt']

#     @property
#     def processed_file_names(self):
#         return 'data.pt'

#     def download(self):
#         for name in self.raw_file_names:
#             download_url(f'{self.url}/{self.name}/{name}', self.raw_dir)

#     def process(self):
#         with open(self.raw_paths[0], 'r') as f:
#             data = f.read().split('\n')[1:-1]
#             x = [[float(v) for v in r.split('\t')[1].split(',')] for r in data]
#             x = torch.tensor(x, dtype=torch.float)

#             y = [int(r.split('\t')[2]) for r in data]
#             y = torch.tensor(y, dtype=torch.long)

#         with open(self.raw_paths[1], 'r') as f:
#             data = f.read().split('\n')[1:-1]
#             data = [[int(v) for v in r.split('\t')] for r in data]
#             edge_index = torch.tensor(data, dtype=torch.long).t().contiguous()
#             edge_index = to_undirected(edge_index)
#             edge_index, _ = coalesce(edge_index, None, x.size(0), x.size(0))

#         data = Data(x=x, edge_index=edge_index, y=y)
#         data = data if self.pre_transform is None else self.pre_transform(data)
#         torch.save(self.collate([data]), self.processed_paths[0])

#     def __repr__(self):
#         return '{}()'.format(self.name)


def DataLoader(name, split):

    root_path = '/mnt/jiahanli/datasets/gprgnn'

    if name in ['chameleon','squirrel']:
        dataset = WikipediaNetwork(root=root_path, name=name, geom_gcn_preprocess=True)
    elif name in ['cornell', 'texas', 'wisconsin']:
        dataset = WebKB(root_path, name=name)
    elif name == 'film':
        dataset = Actor(root=root_path)
    else:
        raise ValueError(f'dataset {name} not supported in dataloader')


    data = dataset[0]
    if name in ['chameleon', 'squirrel']:
        splits_file = np.load(f'{root_path}/{name}/geom_gcn/raw/{name}_split_0.6_0.2_{split}.npz')
    if name in ['cornell', 'texas', 'wisconsin']:
        splits_file = np.load(f'{root_path}/{name}/raw/{name}_split_0.6_0.2_{split}.npz')
    if name == 'film':
        splits_file = np.load(f'{root_path}/raw/{name}_split_0.6_0.2_{split}.npz')

    train_mask = splits_file['train_mask']
    val_mask = splits_file['val_mask']
    test_mask = splits_file['test_mask']

    data.train_mask = torch.tensor(train_mask, dtype=torch.bool)
    data.val_mask = torch.tensor(val_mask, dtype=torch.bool)
    data.test_mask = torch.tensor(test_mask, dtype=torch.bool)

    return dataset, data
