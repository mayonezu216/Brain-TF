import os
import os.path as osp
import torch

import glob
import numpy as np
from torch_geometric.data import Dataset, Data, download_url
from tqdm import tqdm
from scipy import sparse
import pandas as pd
class HCPDataset(Dataset):
    def __init__(self, root = '.', transform=None, pre_transform=None):
        '''
        root: where the dataset should be stored. This folder is split into raw_dir (download dataset)
        and processed_dir (processed data).
        '''
        # self.embeddings = embeddings
        self.root = root
        self.filenames = glob.glob(root + '/*.txt')
        self.filenames = sorted(self.filenames)
        self.graph_list = self.filenames
        self.all_info = pd.read_excel("/fast/beidi/GraphCaps/HCPA_demographics.xlsx",
                         sheet_name = "hcp-a_for_stats",
                         decimal='.')
        # print(self.filenames)
        # print(self.filenames )
        super(HCPDataset,self).__init__(root,transform, pre_transform)

    @property
    def raw_file_names(self):
        '''If this file exists in raw_dir, the download function is not implemented.
        '''
        return '1.txt'

    @property
    def processed_file_names(self):
        return 'not_implemented.pt'

    def download(self):
        # Download to `self.raw_dir`.
        # path = download_url(url, self.raw_dir)
        pass

    def process(self):
        if not os.path.exists(os.path.join(self.root,'saved_graph')):
            os.makedirs(os.path.join(self.root,'saved_graph'))

        for i in range(len(self.graph_list)): 
            name = self.graph_list[i]
            embeddings = np.loadtxt(name)
            # print(embeddings.shape)
            # assert 2==3
            node_feats = self._get_node_features(embeddings)
            edge_index = self._get_adjacency_info(embeddings)
            edge_feats = self._get_edge_features(embeddings)

            # row = self.all_info[self.all_info["src_subject_id"] == i].index.tolist()[0]  
            sex = self.all_info["sex"].map({"M":0,"F":1})
    

            self.label =torch.tensor(sex[i])

            
            data = Data(x = node_feats,
                        edge_index = edge_index,
                        edge_attr = edge_feats,

                        y = self.label
                        )
            # os.makedirs(os.path.join(self.real_root,'saved_graph'))
            processed_dir = os.path.join(self.root,'saved_graph')
            save_name = name.split('/')[-1].split('_')[0]
    
            # if not os.path.exists()
            torch.save(data,osp.join(processed_dir,
                                f'graph_{save_name}.pt'))


        # if self.pre_filter is not None and not self.pre_filter(data):
        #     continue

        # if self.pre_transform is not None:
        #     data = self.pre_transform(data)



    def _get_node_features(self,data):
        return torch.tensor(data,dtype = torch.float)

    def _get_edge_features(self,data):
        row,col = np.diag_indices_from(data)

        data[row,col] = 0
        adj = data > 0.5
        
        adj = sparse.csr_matrix(adj).tocoo()
        # print(adj.shape)
        edge_index = self._get_adjacency_info(data)
        # print(edge_index.size())
        
        edge_feat = []
        for i in range(edge_index.size(1)):
            edge_feat.append(data[edge_index[0,i],edge_index[1,i]])
        # print(torch.tensor(edge_feat,dtype = torch.long).size())
        
        # # edge_feat = np.reshape(adj.data,[adj.data.shape,1])

        
        # edge_feat = data @ adj

        # print(torch.tensor(edge_feat,dtype = torch.long).size())
        # edge_feat = np.reshape(adj.data,[adj.data.shape,1])
        return torch.tensor(edge_feat,dtype = torch.float)


    def _get_adjacency_info(self,data):
        row,col = np.diag_indices_from(data)

        data[row,col] = 0
        #print(data.shape)
        adj = data > 0.5

        adj = sparse.csr_matrix(adj).tocoo()
        # print(adj)
        row = torch.from_numpy(adj.row.astype(np.int64)).to(torch.long)
        col = torch.from_numpy(adj.col.astype(np.int64)).to(torch.long)
        edge_index = torch.stack([row, col], dim=0)
        # print(edge_index)
        # return torch.tensor(edge_index,dtype = torch.long).clone().detach()

        return torch.as_tensor(edge_index, dtype=torch.long)
    # def _get_labels(self,data):


    def len(self):
        # return len(self.processed_file_names)
        return len(self.filenames)



    def get(self, idx):
        processed_dir = os.path.join(self.root, 'saved_graph')
        graph_name = self.graph_list[idx].split('.')[0].split('/')[-1].split('_')[0]
        data = torch.load(osp.join(processed_dir, f'graph_{graph_name}.pt'))
        return data


# embeddings = np.random.rand(30,2048)
# dataset = UrineDataset(root = 'data/')
# print(dataset[0].edge_index.t())
# print(dataset[0].x)
# print(dataset[0].edge_attr)
# print(dataset[0].

class HCP_ave_Dataset(Dataset):
    def __init__(self, root = '.', transform=None, pre_transform=None):
        '''
        root: where the dataset should be stored. This folder is split into raw_dir (download dataset)
        and processed_dir (processed data).
        '''
        # self.embeddings = embeddings
        self.root = root
        self.filenames = glob.glob(root + '/*.txt')
        self.filenames = sorted(self.filenames)
        self.graph_list = self.filenames
        self.all_info = pd.read_excel("/fast/beidi/GraphCaps/HCPA_demographics.xlsx",
                         sheet_name = "hcp-a_for_stats",
                         decimal='.')
        # print(self.filenames)
        # print(self.filenames )
        super(HCP_ave_Dataset,self).__init__(root,transform, pre_transform)

    @property
    def raw_file_names(self):
        '''If this file exists in raw_dir, the download function is not implemented.
        '''
        return '1.txt'

    @property
    def processed_file_names(self):
        return 'not_implemented.pt'

    def download(self):
        # Download to `self.raw_dir`.
        # path = download_url(url, self.raw_dir)
        pass

    def process(self):
        if not os.path.exists(os.path.join(self.root,'saved_graph')):
            os.makedirs(os.path.join(self.root,'saved_graph'))
        
        for i in range(len(self.graph_list)): 
            name = self.graph_list[i]
            if i==0:
                embeddings_ave = np.loadtxt(name)[:,:,np.newaxis]
            else:
                embeddings_ave = np.concatenate((embeddings_ave,embeddings),2)
            embeddings = np.loadtxt(name)[:,:,np.newaxis]
            

            print(embeddings_ave.shape)
        embeddings = np.mean(embeddings_ave,axis=2)
            # assert 2==3
        node_feats = self._get_node_features(embeddings)
        edge_index = self._get_adjacency_info(embeddings)
        edge_feats = self._get_edge_features(embeddings)

        # row = self.all_info[self.all_info["src_subject_id"] == i].index.tolist()[0]  
        sex = self.all_info["sex"].map({"M":0,"F":1})


        self.label =torch.tensor(sex[i])

        
        data = Data(x = node_feats,
                    edge_index = edge_index,
                    edge_attr = edge_feats,

                    y = self.label
                    )
        # os.makedirs(os.path.join(self.real_root,'saved_graph'))
        processed_dir = os.path.join(self.root,'saved_graph')
        save_name = name.split('/')[-1].split('_')[0]

        # if not os.path.exists()
        torch.save(data,osp.join(processed_dir,
                            f'graph_{save_name}.pt'))


        # if self.pre_filter is not None and not self.pre_filter(data):
        #     continue

        # if self.pre_transform is not None:
        #     data = self.pre_transform(data)



    def _get_node_features(self,data):
        return torch.tensor(data,dtype = torch.float)

    def _get_edge_features(self,data):
        row,col = np.diag_indices_from(data)

        data[row,col] = 0
        adj = data > 0.5
        
        adj = sparse.csr_matrix(adj).tocoo()
        # print(adj.shape)
        edge_index = self._get_adjacency_info(data)
        # print(edge_index.size())
        
        edge_feat = []
        for i in range(edge_index.size(1)):
            edge_feat.append(data[edge_index[0,i],edge_index[1,i]])
        # print(torch.tensor(edge_feat,dtype = torch.long).size())
        
        # # edge_feat = np.reshape(adj.data,[adj.data.shape,1])

        
        # edge_feat = data @ adj

        # print(torch.tensor(edge_feat,dtype = torch.long).size())
        # edge_feat = np.reshape(adj.data,[adj.data.shape,1])
        return torch.tensor(edge_feat,dtype = torch.float)


    def _get_adjacency_info(self,data):
        row,col = np.diag_indices_from(data)

        data[row,col] = 0
        #print(data.shape)
        adj = data > 1

        adj = sparse.csr_matrix(adj).tocoo()
        # print(adj)
        row = torch.from_numpy(adj.row.astype(np.int64)).to(torch.long)
        col = torch.from_numpy(adj.col.astype(np.int64)).to(torch.long)
        edge_index = torch.stack([row, col], dim=0)
        # print(edge_index)
        # return torch.tensor(edge_index,dtype = torch.long).clone().detach()

        return torch.as_tensor(edge_index, dtype=torch.long)
    # def _get_labels(self,data):


    def len(self):
        # return len(self.processed_file_names)
        return len(self.filenames)



    def get(self, idx):
        idx = 0
        processed_dir = os.path.join(self.root, 'saved_graph')
        graph_name = self.graph_list[idx].split('.')[0].split('/')[-1].split('_')[0]
        data = torch.load(osp.join(processed_dir, f'graph_{graph_name}.pt'))
        return data
