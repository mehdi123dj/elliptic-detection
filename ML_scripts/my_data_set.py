# -*- coding: utf-8 -*-

import os
from typing import Callable, List, Optional
import torch
from torch_geometric.data import Data,InMemoryDataset,download_url, extract_zip

class MyEllipticBitcoinDataset(InMemoryDataset):
    
    url = 'https://data.pyg.org/datasets/elliptic'
    def __init__(
                self,
                root: str, 
                split = 'train',
                transform: Optional[Callable] = None,
                pre_transform: Optional[Callable] = None,
                ):
        
        super().__init__(root, transform, pre_transform)
        
        if split == 'train':
            self.data, self.slices = torch.load(self.processed_paths[0])
        elif split == 'val':
            self.data, self.slices = torch.load(self.processed_paths[1])
        elif split == 'test':
            self.data, self.slices = torch.load(self.processed_paths[2])
            
    @property
    def raw_file_names(self) -> List[str]:
        return [
            'elliptic_txs_features.csv',
            'elliptic_txs_edgelist.csv',
            'elliptic_txs_classes.csv',
        ]
    
    @property
    def processed_file_names(self) -> str:
        return ['train.pt', 'val.pt', 'test.pt']

    def download(self):
        for file_name in self.raw_file_names:
            path = download_url(f'{self.url}/{file_name}.zip', self.raw_dir)
            extract_zip(path, self.raw_dir)
            os.remove(path)

    def process(self):
        import pandas as pd
        
        
        df_features = pd.read_csv(self.raw_paths[0], header=None)
        columns = {0: 'txId', 1: 'time_step'}
        df_features = df_features.rename(columns=columns)
        
        df_edges = pd.read_csv(self.raw_paths[1])
        
        df_classes = pd.read_csv(self.raw_paths[2])
        mapping = {'unknown': 2, '1': 1, '2': 0}
        df_classes['class'] = df_classes['class'].map(mapping)
        

        config = {'train' : (1,33),
                  'val'   : (33,41),
                  'test'  : (41,50)}
    
        
        for s in range(len(config)):
            t1,t2 = list(config.values())[s]
            data_list = []
            for timestep in range(t1,t2):
                node_idx = list(df_features[df_features['time_step'] == timestep].index)

                x = torch.from_numpy(df_features.iloc[node_idx].loc[:, 2:].values).to(torch.float)

                # There exists 3 different classes in the dataset:
                # 0=licit,  1=illicit, 2=unknown

                y = torch.from_numpy(df_classes.iloc[node_idx]['class'].values)

                mapping = {idx: i for i, idx in enumerate(df_features.iloc[node_idx]['txId'].values)}
                df_E = df_edges[df_edges['txId1'].isin(df_features['txId']) & df_edges['txId2'].isin(df_features['txId'])]
                df_E['txId1'] = df_E['txId1'].map(mapping)
                df_E['txId2'] = df_E['txId2'].map(mapping)
                df_E = df_E.dropna()

                edge_index = torch.from_numpy(df_E.values).t().long()
                data_list.append(Data(x=x, edge_index=edge_index, y=y))

            if self.pre_filter is not None:
                data_list = [data for data in data_list if self.pre_filter(data)]

            if self.pre_transform is not None:
                data_list = [self.pre_transform(data) for data in data_list]

            #cls = data_list[0].__class__ 
            data, slices = self.collate(data_list)
            torch.save((data, slices), self.processed_paths[s])


    @property
    def num_classes(self) -> int:
        return 2

