from scipy.io import mmread
import numpy as np
import pandas as pd
from collections import Counter
import torch
from torch.utils.data import TensorDataset,DataLoader
from torch.utils.data import random_split
import os
from torch.nn import functional as F

EPS = 1e-12


def build_pt_data(mtx_file, meta_file, save_path, train_pct = 0.8, 
                  gene_by_cell = True, group_indexes = dict(), 
                  label_cols = ['group'], disc_labels = list()):
    '''
    Read a single cell data and convert data to .pt files for pytorch input
    Meanwhile split training set and validation set
    
    Parameters:
    -----------
    mtx_file: str 
        Path to the matrix file
    meta_file: str
        Path to the metadata file
    save_path: str
        Path for saving output files
    train_pct: float between [0,1]
        Percentage for data that will be sampled into training set, 
        other data will form a validataion set
    gene_by_cell: boolean
        Whether the matrix is gene*by*cell, if False, the matrix will be treated as cell*gene
    group_indexes: dictionary of dictionaries
        The mapping relationship between group name and a group index number,
        the key of outer dict should be a group name, 
        and value is the mapping relationship of that group
        If None, group name will be automatically assigned to range(1,#group)
        and will be ordered by group size
    label_cols: list of strs
        Which column/columns will be treated as labels
    disc_labels: list of strs
        Which column/columns will be treated as discrete labels, 
        If empty, only dtype=='object' dims will be disc
    '''
    # read data
    mtx = mmread(mtx_file)
    metadata = pd.read_csv(meta_file)
    # convert matrix to tensor
    if gene_by_cell:
        mtx = mtx.transpose()
    values = mtx.data
    indices = np.vstack((mtx.row, mtx.col))
    i = torch.LongTensor(indices)
    v = torch.FloatTensor(values)
    shape = mtx.shape
    mtx_torch = torch.sparse.FloatTensor(i, v, torch.Size(shape))
    # convert metadata to tensor, only use labeled columns, 
    # and if a columns is catagorical, map it to indexes
    label_torch_names = []
    for l in label_cols:
        l_data = metadata[l]
        is_discrete = l in disc_labels or l in group_indexes.keys() or l_data.dtype in ['object']
        if not is_discrete:
            label_torch_names.append(l)
        else:
            counter = Counter(l_data).most_common()
            if l in group_indexes.keys():
                group_index = group_indexes[l]
            else:
                counter = Counter(l_data).most_common()
                group_index = {key:idx for idx,key in enumerate([c[0] for c in counter])}
                group_indexes[l] = group_index
            metadata[l+'.index'] = [group_index[i] for i in l_data]
            label_torch_names.append(l+'.index')
            print('convert column ['+l+'] to index, mapping rule:')
            print(group_index)
    label_torch = metadata[label_torch_names]
    # split dataset to training and validation
    mtx_torch = mtx_torch.to_dense()
    label_torch = torch.tensor(np.array(label_torch))
    print('matrix size: '+ str(mtx_torch.shape))
    print('metadata size: '+ str(label_torch.shape))
    dataset_torch = TensorDataset(mtx_torch,label_torch)
    train_num = int(len(dataset_torch)*train_pct)
    train,val = random_split(dataset_torch,[train_num,len(dataset_torch)-train_num])
    # save torch datasets
    torch.save(dataset_torch, os.path.join(save_path, 'whole_dataset.pt'))
    torch.save(train, os.path.join(save_path, 'train_dataset.pt'))
    torch.save(val, os.path.join(save_path, 'val_dataset.pt'))
    # load: `gi = np.load(file).item()`
    np.save(os.path.join(save_path, 'disc_labels_mapping.npy'), group_indexes) 
    # load: `file.read().splitlines()`
    file = open(os.path.join(save_path, 'label_names.txt'),'w')
    file.writelines([str(i)+'\n' for i in label_cols])
    file.close()


def sample_normal(mean, logvar, training, use_cuda):
    """
    Samples from a normal distribution using the reparameterization trick.

    Parameters
    ----------
    mean : torch.Tensor
        Mean of the normal distribution. Shape (N, D) where D is dimension
        of distribution.

    logvar : torch.Tensor
        Diagonal log variance of the normal distribution. Shape (N, D)
    """
    if training:
        std = torch.exp(0.5 * logvar)
        eps = torch.zeros(std.size()).normal_()
        if use_cuda:
            eps = eps.cuda()
        return mean + std * eps
    else:
        # Reconstruction mode
        return mean

def sample_gumbel_softmax(alpha,temperature, training, use_cuda):
    """
    Samples from a gumbel-softmax distribution using the reparameterization
    trick.

    Parameters
    ----------
    alpha : torch.Tensor
        Parameters of the gumbel-softmax distribution. Shape (N, D)
    """
    if training:
        # Sample from gumbel distribution
        unif = torch.rand(alpha.size())
        if use_cuda:
            unif = unif.cuda()
        gumbel = -torch.log(-torch.log(unif + EPS) + EPS)
        # Reparameterize to create gumbel softmax sample
        log_alpha = torch.log(alpha + EPS)
        logit = (log_alpha + gumbel) / temperature
        return F.softmax(logit, dim=1)
    else:
        # In reconstruction mode, pick most likely sample
        _, max_alpha = torch.max(alpha, dim=1)
        one_hot_samples = torch.zeros(alpha.size())
        # On axis 1 of one_hot_samples, scatter the value 1 at indices
        # max_alpha. Note the view is because scatter_ only accepts 2D
        # tensors.
        one_hot_samples.scatter_(1, max_alpha.view(-1, 1).data.cpu(), 1)
        if use_cuda:
            one_hot_samples = one_hot_samples.cuda()
        return one_hot_samples

def pairwise_distances(x, y=None):
        '''
        Input: x is a Nxd matrix
               y is an optional Mxd matirx
        Output: dist is a NxM matrix where dist[i,j] is the square norm between x[i,:] and y[j,:]
                if y is not given then use 'y=x'.
        i.e. dist[i,j] = ||x[i,:]-y[j,:]||^2
        '''
        x_norm = (x**2).sum(1).view(-1, 1)
        if y is not None:
            y_t = torch.transpose(y, 0, 1)
            y_norm = (y**2).sum(1).view(1, -1)
        else:
            y_t = torch.transpose(x, 0, 1)
            y_norm = x_norm.view(1, -1)

        dist = x_norm + y_norm - 2.0 * torch.mm(x, y_t)
        # Ensure diagonal is zero if x=y
        # if y is None:
        #     dist = dist - torch.diag(dist.diag)
        return torch.clamp(dist, 0.0, np.inf)


