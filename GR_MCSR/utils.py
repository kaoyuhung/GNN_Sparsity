import pickle
import torch
import numpy as np
import pandas as pd
import scipy.sparse as sp
import math
from scipy.sparse import linalg

class StandardScaler():
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def transform(self, data):
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        return (data * self.std) + self.mean

class StandardScalerLocal():
    def __init__(self, init_data, device):
        self.mean = np.mean(init_data, axis=0)
        self.std = np.std(init_data, axis=0)
        self.mean_torch = torch.Tensor(self.mean)[:, None].to(device)
        self.std_torch = torch.Tensor(self.std)[:, None].to(device)
    
    def transform(self, data):
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        return (data * self.std_torch) + self.mean_torch
    
def getTimestamp(data):
    num_samples, num_nodes = data.shape
    time_ind = (data.index.values - data.index.values.astype("datetime64[D]")) / np.timedelta64(1, "D")
    time_in_day = np.tile(time_ind, [num_nodes,1]).transpose((1, 0))
    return time_in_day

def getDayTimestamp(data):
    # 288 timeslots each day for dataset has 5 minutes time interval.
    df = pd.DataFrame({'timestamp':data.index.values})
    df['weekdaytime'] = df['timestamp'].dt.weekday * 288 + (df['timestamp'].dt.hour * 60 + df['timestamp'].dt.minute)//5
    df['weekdaytime'] = df['weekdaytime'] / df['weekdaytime'].max()
    num_samples, num_nodes = data.shape
    time_ind = df['weekdaytime'].values
    time_ind_node = np.tile(time_ind, [num_nodes,1]).transpose((1, 0))
    return time_ind_node

def getDayTimestamp_(start, end, freq, num_nodes):
    # 288 timeslots each day for dataset has 5 minutes time interval.
    df = pd.DataFrame({'timestamp':pd.date_range(start=start, end=end, freq=freq)})
    df['weekdaytime'] = df['timestamp'].dt.weekday * 288 + (df['timestamp'].dt.hour * 60 + df['timestamp'].dt.minute)//5
    df['weekdaytime'] = df['weekdaytime'] / df['weekdaytime'].max()
    time_ind = df['weekdaytime'].values
    time_ind_node = np.tile(time_ind, [num_nodes, 1]).transpose((1, 0))
    return time_ind_node

def masked_mae(preds, labels, null_val=0.0):
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = (labels!=null_val)
    mask = mask.float()
    mask /=  torch.mean((mask))
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    loss = torch.abs(preds-labels)
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return torch.mean(loss)

# DCRNN
def masked_mae_loss(y_pred, y_true):
    mask = (y_true != 0).float()
    mask /= mask.mean()
    loss = torch.abs(y_pred - y_true)
    loss = loss * mask
    # trick for nans: https://discuss.pytorch.org/t/how-to-set-nan-in-tensor-to-0/3918/3
    loss[loss != loss] = 0
    return loss.mean()

def load_pickle(pickle_file):
    try:
        with open(pickle_file, 'rb') as f:
            pickle_data = pickle.load(f)
    except UnicodeDecodeError as e:
        with open(pickle_file, 'rb') as f:
            pickle_data = pickle.load(f, encoding='latin1')
    except Exception as e:
        print('Unable to load data ', pickle_file, ':', e)
        raise
    return pickle_data

def print_params(model):
    # print trainable params
    param_count = 0
    print('Trainable parameter list:')
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(name, param.shape, param.numel())
            param_count += param.numel()
    print(f'\n In total: {param_count} trainable parameters. \n')
    return

def get_normalized_adj(A):
    """
    Returns the degree normalized adjacency matrix.
    """
    A = A + np.diag(np.ones(A.shape[0], dtype=np.float32))
   
    D = np.array(np.sum(A, axis=1)).reshape((-1,))
    D[D <= 10e-5] = 10e-5    # Prevent infs

    diag = np.reciprocal(np.sqrt(D))
    
    A_wave = np.multiply(np.multiply(diag.reshape((-1, 1)), A),
                         diag.reshape((1, -1)))
    return A_wave

def get_sparsity(mat):
    print(mat.shape)
    print('nonzero: ', torch.count_nonzero(mat))
    if len(mat.shape) == 2:
        print('total: ', mat.shape[0]*mat.shape[1])
        print('ratio: ', torch.count_nonzero(mat)/(mat.shape[0]*mat.shape[1]))
    else:
        print('total: ', mat.shape[0]*mat.shape[1]*mat.shape[2])
        print('ratio: ', torch.count_nonzero(mat)/(mat.shape[0]*mat.shape[1]*mat.shape[2]))

def calculate_normalized_laplacian(adj):
    """
    # L = D^-1/2 (D-A) D^-1/2 = I - D^-1/2 A D^-1/2
    # D = diag(A 1)
    :param adj:
    :return:
    """
    adj = sp.coo_matrix(adj)
    d = np.array(adj.sum(1))
    d_inv_sqrt = np.power(d, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    normalized_laplacian = sp.eye(adj.shape[0]) - adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()
    return normalized_laplacian


def calculate_random_walk_matrix(adj_mx):
    adj_mx = sp.coo_matrix(adj_mx)
    d = np.array(adj_mx.sum(1))
    d_inv = np.power(d, -1).flatten()
    d_inv[np.isinf(d_inv)] = 0.
    d_mat_inv = sp.diags(d_inv)
    random_walk_mx = d_mat_inv.dot(adj_mx).tocoo()
    return random_walk_mx


def calculate_reverse_random_walk_matrix(adj_mx):
    return calculate_random_walk_matrix(np.transpose(adj_mx))


def calculate_scaled_laplacian(adj_mx, lambda_max=2, undirected=True):
    if undirected:
        adj_mx = np.maximum.reduce([adj_mx, adj_mx.T])
    L = calculate_normalized_laplacian(adj_mx)
    if lambda_max is None:
        lambda_max, _ = linalg.eigsh(L, 1, which='LM')
        lambda_max = lambda_max[0]
    L = sp.csr_matrix(L)
    M, _ = L.shape
    I = sp.identity(M, format='csr', dtype=L.dtype)
    L = (2 / lambda_max * L) - I
    return L.astype(np.float32)

def graph_reordering(adj_mat, data, data_time, simfunc='Jaccard'):
    print('Graph Reordering...')
    def jaccard_sim(a, b):
        s1, s2 = set(np.nonzero(a)[0]), set(np.nonzero(b)[0])
        intersection = len(s1 & s2)
        union = len(s1 | s2)
        return intersection / union
    
    def cos_sim(a, b):
        a, b = np.where(a > 0, 1, 0), np.where(b > 0, 1, 0)
        return np.dot(a,b) / np.sqrt(np.count_nonzero(a) * np.count_nonzero(b))

    def cos_sim2(a, b):
        a, b = np.where(a > 0, 0, 1), np.where(b > 0, 0, 1)
        return np.dot(a,b) / math.sqrt(np.sum(a) * np.sum(b))

    def new_sim(a, b, alpha=1.2):
        a, b = np.where(a > 0, 0, 1), np.where(b > 0, 0, 1)
        s1, s2 = set(np.nonzero(a)[0]), set(np.nonzero(b)[0])
        intersection = len(s1 & s2)
        union = len(s1 | s2)
        return max(0, intersection - alpha * (union-intersection))

    def union(a ,b):
        return np.where((a+b) != 0, 1, 0)

    undirected_adj_mat = np.zeros(adj_mat.shape)

    for i in range(adj_mat.shape[0]):
        for j in range(adj_mat.shape[1]):
            if adj_mat[i][j] != 0:
                undirected_adj_mat[i][j], undirected_adj_mat[j][i] = 1, 1

    MAX = 32
    mask = np.zeros(adj_mat.shape[0], dtype=bool)
    num_group = int(np.ceil(adj_mat.shape[0] / MAX))
    mapping_list = []
    last = -1
    for i in range(num_group):
        ungrouped_idx = np.argwhere(mask==False)
        ungrouped_idx = ungrouped_idx.reshape(ungrouped_idx.shape[0])
        if i == num_group-1:
            mapping_list += list(ungrouped_idx)
            break
        if i == 0:
            choose = np.where(undirected_adj_mat[0] != 0, 1, 0)
        else:
            choose = np.where(undirected_adj_mat[last] != 0, 1, 0)
        ungrouped_idx = set(ungrouped_idx)
        for _ in range(MAX+1):
            M, M_idx = -1, -1
            for idx in ungrouped_idx:
                if simfunc == 'Jaccard':    
                    score = jaccard_sim(choose, undirected_adj_mat[idx])
                elif simfunc == 'cosine2':
                    score = cos_sim2(choose, undirected_adj_mat[idx])
                elif simfunc == 'new_sim':
                    score = new_sim(choose, undirected_adj_mat[idx])
                else:
                    score = cos_sim(choose, undirected_adj_mat[idx])
                if score > M:
                    M , M_idx = score, idx
            if _ == MAX:
                last = M_idx
                break
            mapping_list.append(M_idx)
            mask[M_idx] = True
            ungrouped_idx.remove(M_idx)
            choose = union(choose,  np.where(undirected_adj_mat[M_idx] != 0, 1, 0))
    
    new_adj_mat = np.zeros(adj_mat.shape)
    index_to = {k: v for v, k in enumerate(mapping_list)}
    
    for i in range(adj_mat.shape[0]):
        for j in range(adj_mat.shape[1]):
            if adj_mat[i][j] != 0:
                new_adj_mat[index_to[i]][index_to[j]] = adj_mat[i][j]

    return new_adj_mat, data[:, mapping_list], data_time[:, mapping_list]
