import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse as sparse
import math
import argparse
import pandas as pd
import matplotlib.cm as cm
import random
from utils import get_normalized_adj

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, choices=['METRLA', 'PEMSBAY', 'PEMS03', 'PEMS04', 'PEMS07', 'PEMS08', 'CORA'], default='METRLA')
parser.add_argument('--show', type=eval, choices=[True, False], default=False)
parser.add_argument('--version', type=str, choices=['origin', 'degree', 'v1', 'v2', 'v3', 'v4', 'v5'], default='v3')
parser.add_argument('--sim_func', type=str, choices=['Jaccard', 'cosine', 'cosine2', 'new_sim'], default='jaccard')
parser.add_argument('--blockSZ_diff', type=eval, choices=[True, False], default=False)
parser.add_argument('--blockSZ_graph_diff', type=eval, choices=[True, False], default=False)
parser.add_argument('--blockSZ_CORA_diff', type=eval, choices=[True, False], default=False)
parser.add_argument('--len', type=int, default=32)
parser.add_argument('--toMCSR', type=eval, choices=[True, False], default=False)
parser.add_argument('--MCSR_compare', type=eval, choices=[True, False], default=False)
parser.add_argument('--proj_data', type=eval, choices=[True, False], default=False)
parser.add_argument('--check_graph', type=eval, choices=[True, False], default=False)
opt = parser.parse_args()
MAX = opt.len

class Modified_CSR():
    def __init__(self, len) -> None:
        self.len = len
        self.data_arr = []
        self.col_idx = []
        self.row_ptr = [0]
        self.row_len = None

    def transform(self, adj_mat):
        self.__init__(self.len)
        cur = 0
        self.row_len = adj_mat.shape[0]
        for i in range(0, adj_mat.shape[0], self.len):
            for j in range(adj_mat.shape[1]):
                for k in range(i, min(i + self.len, self.row_len)):
                   if adj_mat[k][j] != 0:
                       if i + self.len <= self.row_len:
                            self.data_arr.extend([adj_mat[k][j] for k in range(i, i + self.len)])
                       else:
                            self.data_arr.extend([adj_mat[k][j] for k in range(i, self.row_len)] + [0 for _ in range(i + self.len - self.row_len)])
                       self.col_idx.append(j)
                       cur+=1
                       break
            self.row_ptr.append(cur)

    def get_statistics(self):
        return [len(self.row_ptr), len(self.col_idx), len(self.data_arr)]

def get_sparsity(mat):
    print(mat.shape)
    print('nonzero: ', np.count_nonzero(mat))
    if len(mat.shape) == 2:
        print('total: ', mat.shape[0]*mat.shape[1])
        print('ratio: ', np.count_nonzero(mat)/(mat.shape[0]*mat.shape[1]))
    else:
        print('total: ', mat.shape[0]*mat.shape[1]*mat.shape[2])
        print('ratio: ', np.count_nonzero(mat)/(mat.shape[0]*mat.shape[1]*mat.shape[2]))


def check_correctness(adj_mat, new_adj_mat):
    colsum = np.count_nonzero(adj_mat, axis=0)
    rowsum = np.count_nonzero(adj_mat, axis=1)
    degree = colsum + rowsum
    print(np.sort(degree))
    colsum = np.count_nonzero(new_adj_mat, axis=0)
    rowsum = np.count_nonzero(new_adj_mat, axis=1)
    degree = colsum + rowsum
    print(np.sort(degree))


def jaccard_sim(a, b):
    s1, s2 = set(np.nonzero(a)[0]), set(np.nonzero(b)[0])
    intersection = len(s1 & s2)
    union = len(s1 | s2)
    return intersection / union

def cos_sim(a, b):
    a, b = np.where(a > 0, 1, 0), np.where(b > 0, 1, 0)
    return np.dot(a,b) / math.sqrt(np.count_nonzero(a) * np.count_nonzero(b))

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

def reordering_by_degree(adj_mat):
    colsum = np.count_nonzero(adj_mat, axis=0)
    rowsum = np.count_nonzero(adj_mat, axis=1)
    degree = colsum + rowsum
    argsort = np.argsort(-degree)
    index_to = {k: v for v, k in enumerate(argsort)}

    new_adj_mat = np.zeros(adj_mat.shape)
    for i in range(adj_mat.shape[0]):
        for j in range(adj_mat.shape[1]):
            if adj_mat[i][j] != 0:
                new_adj_mat[index_to[i]][index_to[j]] = adj_mat[i][j]
    
    return new_adj_mat

def reordering_by_sim(adj_mat, simfunc='Jaccard'):
    mask = np.zeros(adj_mat.shape[0], dtype=bool)
    group_num = int(np.ceil(adj_mat.shape[0] / MAX))
    mapping_list = []
    for _ in range(group_num):
        ungrouped_idx = np.argwhere(mask==False)
        choose, ungrouped_idx = np.where(adj_mat[ungrouped_idx[0][0]] != 0, 1, 0), ungrouped_idx.reshape(ungrouped_idx.shape[0])
        scores = []
        for idx in ungrouped_idx:
            if simfunc == 'Jaccard':    
                scores.append(jaccard_sim(choose, adj_mat[idx]))
            elif simfunc == 'cosine2':
                scores.append(cos_sim2(choose, adj_mat[idx]))
            else:
                scores.append(cos_sim(choose, adj_mat[idx]))
        scores_argsort = np.argsort(-np.array(scores))[:min(MAX, len(ungrouped_idx))]
        for idx in scores_argsort:
            mapping_list.append(ungrouped_idx[idx])
            mask[ungrouped_idx[idx]] = True
    
    new_adj_mat = np.zeros(adj_mat.shape)
    index_to = {k: v for v, k in enumerate(mapping_list)}

    for i in range(adj_mat.shape[0]):
        for j in range(adj_mat.shape[1]):
            if adj_mat[i][j] != 0:
                new_adj_mat[index_to[i]][index_to[j]] = adj_mat[i][j]
    
    return new_adj_mat

def reordering_by_simV2(adj_mat, simfunc='Jaccard'):
    mask = np.zeros(adj_mat.shape[0], dtype=bool)
    num_group = int(np.ceil(adj_mat.shape[0] / MAX))
    mapping_list = []
    for _ in range(num_group):
        ungrouped_idx = np.argwhere(mask==False)
        choose= np.where(adj_mat[ungrouped_idx[0][0]] != 0, 1, 0)
        ungrouped_idx = {val for val in ungrouped_idx.reshape(ungrouped_idx.shape[0])}
        if _ == num_group - 1:
            mapping_list += list(ungrouped_idx)
            break
        for _ in range(MAX):
            M, M_idx = -1, -1
            for idx in ungrouped_idx:
                if simfunc == 'Jaccard':    
                    score = jaccard_sim(choose, adj_mat[idx])
                elif simfunc == 'cosine2':
                    score = cos_sim2(choose, adj_mat[idx])
                elif simfunc == 'new_sim':
                    score = new_sim(choose, adj_mat[idx])
                else:
                    score = cos_sim(choose, adj_mat[idx])
                if score > M:
                    M , M_idx = score, idx
            mapping_list.append(M_idx)
            mask[M_idx] = True
            ungrouped_idx.remove(M_idx)
            choose = union(choose,  adj_mat[M_idx])
            
    new_adj_mat = np.zeros(adj_mat.shape)
    index_to = {k: v for v, k in enumerate(mapping_list)}
    
    for i in range(adj_mat.shape[0]):
        for j in range(adj_mat.shape[1]):
            if adj_mat[i][j] != 0:
                new_adj_mat[index_to[i]][index_to[j]] = adj_mat[i][j]

    return new_adj_mat

def reordering_by_simV3(adj_mat, simfunc='Jaccard'):
    undirected_adj_mat = np.zeros(adj_mat.shape)

    for i in range(adj_mat.shape[0]):
        for j in range(adj_mat.shape[1]):
            if adj_mat[i][j] != 0:
                undirected_adj_mat[i][j], undirected_adj_mat[j][i] = 1, 1

    mask = np.zeros(adj_mat.shape[0], dtype=bool)
    num_group = int(np.ceil(adj_mat.shape[0] / MAX))
    mapping_list = []
    for _ in range(num_group):
        ungrouped_idx = np.argwhere(mask==False)
        choose =  np.where(undirected_adj_mat[ungrouped_idx[0][0]] != 0, 1, 0)
        ungrouped_idx = set(ungrouped_idx.reshape(ungrouped_idx.shape[0]))
        if _ == num_group - 1:
            mapping_list += list(ungrouped_idx)
            break
        for _ in range(MAX):
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
            mapping_list.append(M_idx)
            mask[M_idx] = True
            ungrouped_idx.remove(M_idx)
            choose = union(choose,  undirected_adj_mat[M_idx])
            
    new_adj_mat = np.zeros(adj_mat.shape)
    index_to = {k: v for v, k in enumerate(mapping_list)}
    
    for i in range(adj_mat.shape[0]):
        for j in range(adj_mat.shape[1]):
            if adj_mat[i][j] != 0:
                new_adj_mat[index_to[i]][index_to[j]] = adj_mat[i][j]

    return new_adj_mat

def reordering_by_simV4(adj_mat, simfunc='Jaccard'):
    undirected_adj_mat = np.zeros(adj_mat.shape)

    for i in range(adj_mat.shape[0]):
        for j in range(adj_mat.shape[1]):
            if adj_mat[i][j] != 0:
                undirected_adj_mat[i][j], undirected_adj_mat[j][i] = 1, 1

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
            choose = union(choose,  undirected_adj_mat[M_idx])
    
    new_adj_mat = np.zeros(adj_mat.shape)
    index_to = {k: v for v, k in enumerate(mapping_list)}
    
    for i in range(adj_mat.shape[0]):
        for j in range(adj_mat.shape[1]):
            if adj_mat[i][j] != 0:
                new_adj_mat[index_to[i]][index_to[j]] = adj_mat[i][j]

    return new_adj_mat

def reordering_by_simV5(adj_mat, simfunc='Jaccard'):
    mask = np.zeros(adj_mat.shape[0], dtype=bool)
    num_group = int(np.ceil(adj_mat.shape[0] / MAX))
    mapping_list = []
    colsum = np.count_nonzero(adj_mat, axis=0)
    rowsum = np.count_nonzero(adj_mat, axis=1)
    degree = colsum + rowsum
    for _ in range(num_group):
        ungrouped_idx = np.argwhere(mask==False)
        ungrouped_idx = ungrouped_idx.reshape(ungrouped_idx.shape[0])
        if _ == num_group - 1:
            mapping_list += list(ungrouped_idx)
            break
        M = -1
        for idx in ungrouped_idx:
            if degree[idx] > M:
                choose = idx
                M = degree[idx]
        choose= np.where(adj_mat[choose] != 0, 1, 0)
        ungrouped_idx = set(ungrouped_idx)
        for _ in range(MAX):
            M, M_idx = -1, -1
            for idx in ungrouped_idx:
                if simfunc == 'Jaccard':    
                    score = jaccard_sim(choose, adj_mat[idx])
                elif simfunc == 'cosine2':
                    score = cos_sim2(choose, adj_mat[idx])
                elif simfunc == 'new_sim':
                    score = new_sim(choose, adj_mat[idx])
                else:
                    score = cos_sim(choose, adj_mat[idx])
                if score > M:
                    M , M_idx = score, idx
            mapping_list.append(M_idx)
            mask[M_idx] = True
            ungrouped_idx.remove(M_idx)
            choose = union(choose,  adj_mat[M_idx])
            
    new_adj_mat = np.zeros(adj_mat.shape)
    index_to = {k: v for v, k in enumerate(mapping_list)}
    
    for i in range(adj_mat.shape[0]):
        for j in range(adj_mat.shape[1]):
            if adj_mat[i][j] != 0:
                new_adj_mat[index_to[i]][index_to[j]] = adj_mat[i][j]

    return new_adj_mat

def plt_matrix(adj_mat, output=False, file_name=None):
    #plt.imshow(adj_mat,  cmap=cm.Greys_r)
    s = sparse.csr_matrix(adj_mat)
    plt.spy(s, markersize=1)
    if output:
         plt.savefig(file_name)
    plt.show()

    
def get_tile(adj_mat):
    i, cnt = 0, 0
    tiled_mat = np.full((adj_mat.shape[0], adj_mat.shape[0], 3), 255)
    while i < adj_mat.shape[0]:
        j = 0
        while j < adj_mat.shape[1]:
            flag = False
            for k in range(min(MAX, adj_mat.shape[0]-i)):
                if adj_mat[i+k][j] != 0:
                    flag = True
                    break
            if flag:
                cnt += 1
                for k in range(min(MAX,adj_mat.shape[0]-i)):
                    tiled_mat[i+k][j][1] = tiled_mat[i+k][j][2] = 0
                    tiled_mat[i+k][j+min(MAX,adj_mat.shape[1]-j)-1][1] = tiled_mat[i+k][j+min(MAX,adj_mat.shape[1]-j)-1][2] = 0
                for k in range(min(MAX,adj_mat.shape[1]-j)):
                    tiled_mat[i][j+k][1] = tiled_mat[i][j+k][2] = 0
                    tiled_mat[i+min(MAX,adj_mat.shape[0]-i)-1][j+k][1] = tiled_mat[i+min(MAX,adj_mat.shape[0]-i)-1][j+k][2] = 0
                j += MAX
            else:
                j += 1
        i+= MAX

    for i in range(adj_mat.shape[0]):
        for j in range(adj_mat.shape[1]):
            if adj_mat[i][j] != 0:
                if tiled_mat[i][j][0] == 255 and tiled_mat[i][j][1] == 0:
                    tiled_mat[i][j][2] = 255
                else:
                    tiled_mat[i][j][0] = tiled_mat[i][j][1] = 0
    
    return [tiled_mat, cnt]

def get_tiling_result(new_adj_mat_list):
    ret = [get_tile(adj_mat)] + [get_tile(new_adj_mat) for new_adj_mat in new_adj_mat_list]
    return ret

def get_blockSZ_diff_result(adj_mat, dataset):
    global MAX
    result, tiling_result = [], []
    new_adj_mat_list = []
    for num in [4, 8, 16, 32]:
        MAX = num
        new_adj_mat_list.append([reordering_by_simV2(adj_mat), reordering_by_simV2(adj_mat, 'new_sim'), reordering_by_simV2(adj_mat, 'cosine2'),
                                 reordering_by_simV3(adj_mat), reordering_by_simV3(adj_mat, 'new_sim'), reordering_by_simV3(adj_mat, 'cosine2'),
                                 reordering_by_simV4(adj_mat), reordering_by_simV4(adj_mat, 'new_sim'), reordering_by_simV4(adj_mat, 'cosine2')])
        tiling_result.append(get_tiling_result(new_adj_mat_list[-1]))
        result.append([element[1] for element in tiling_result[-1]])
    #return np.array(tiling_result, dtype=object)
    tiling_result = np.array(tiling_result, dtype=object)
    tile_mat, result = tiling_result[:,:,0], tiling_result[:,:,1]
    idx = ['4x4','8x8', '16x16', '32x32']
    cols = ['No reordering', 'simV2-jaccard','simV2-new_sim','simV2-cosine2','simV3-jaccard','simV3-new_sim','simV3-cosine2',
            'simV4-jaccard','simV4-new_sim','simV4-cosine2']
    df = pd.DataFrame(result, columns = cols, index = idx)
    if opt.blockSZ_diff:
        df.to_csv('block_diff({}).csv'.format(dataset))
    if opt.blockSZ_graph_diff:
        df.to_csv('./graph_diff/{}_result.csv'.format(dataset))

    versions = ['v2', 'v3', 'v4']
    for k in range(3):
        plt.figure(figsize=(20,20))
        for i in range(4):
            for j in range(1, 5):
                plt.subplot(4, 4, 4*i+j)
                if j == 1:
                    plt.imshow(tile_mat[i][0])
                    plt.ylabel(idx[i], fontsize = 10)
                else:
                    plt.imshow(tile_mat[i][3*k+j-1])
                if i == 3:
                    if j == 1:
                        plt.xlabel(cols[0], fontsize = 9)
                    else:
                        plt.xlabel(cols[3*k+j-1], fontsize = 9)

        plt.tight_layout()
        if opt.blockSZ_diff:
            plt.savefig('blockSZ_diff_{}_{}'.format(dataset,versions[k]))
            plt.show()
        if opt.blockSZ_graph_diff:
            plt.savefig('./graph_diff/{}_{}_result'.format(dataset,versions[k]))
        plt.clf()
        
    return 

def get_SIMD_len_diff(dataset, adj_mat):
    global MAX
    result = []
    origin_size = adj_mat.shape[0] * adj_mat.shape[1]
    CSR = sparse.csc_matrix(adj_mat)
    for num in [2, 4, 8]:
        MAX = num
        new_adj_mat_list = [adj_mat, reordering_by_simV2(adj_mat), reordering_by_simV2(adj_mat, 'new_sim'), reordering_by_simV2(adj_mat, 'cosine2'),
                                 reordering_by_simV3(adj_mat), reordering_by_simV3(adj_mat, 'new_sim'), reordering_by_simV3(adj_mat, 'cosine2'),
                                 reordering_by_simV4(adj_mat), reordering_by_simV4(adj_mat, 'new_sim'), reordering_by_simV4(adj_mat, 'cosine2'),
                                 reordering_by_simV5(adj_mat), reordering_by_simV5(adj_mat, 'new_sim'), reordering_by_simV5(adj_mat, 'cosine2')]
        MCSRs = [Modified_CSR(MAX) for _ in range(len(new_adj_mat_list))]
        for i in range(len(new_adj_mat_list)):
            MCSRs[i].transform(new_adj_mat_list[i])

        result.append(['%5.3f' % ((len(CSR.indptr) + len(CSR.indices) + len(CSR.data)) / origin_size * 100) + '%']+
                      ['%5.3f' % (sum(MCSR.get_statistics()) / origin_size * 100) + '%' for MCSR in MCSRs])
    
    idx = ['len=2','len=4', 'len=8']
    cols = ['CSR', 'No reordering', 
            'simV2-jaccard','simV2-new_sim','simV2-cosine2',
            'simV3-jaccard','simV3-new_sim','simV3-cosine2',
            'simV4-jaccard','simV4-new_sim','simV4-cosine2',
            'simV5-jaccard','simV5-new_sim','simV5-cosine2']
    df = pd.DataFrame(result, columns = cols, index = idx)
    df.to_csv('./simd_diff/{}_result.csv'.format(dataset))
   
if opt.show:
    adj_mat_path = '../data/{}/adj_mat.npy'.format(opt.dataset)
    adj_mat = np.where(get_normalized_adj(np.load(adj_mat_path)) > 0, 1, 0)
    version = opt.version
    print('version: ', version)
    if version[0] == 'v':
        print('sim_func:', opt.sim_func)

    if version == 'origin':
        new_adj_mat = adj_mat
    elif version == 'degree':
        new_adj_mat = reordering_by_degree(adj_mat)
    elif version == 'v1':
        new_adj_mat = reordering_by_sim(adj_mat, opt.sim_func)
    elif version == 'v2':
        new_adj_mat = reordering_by_simV2(adj_mat, opt.sim_func)
    elif version == 'v3':
        new_adj_mat = reordering_by_simV3(adj_mat, opt.sim_func)
    elif version == 'v4':
        new_adj_mat = reordering_by_simV4(adj_mat, opt.sim_func)
    elif version == 'v5':
        new_adj_mat = reordering_by_simV5(adj_mat, opt.sim_func)

    plt_matrix(new_adj_mat)
    tile_mat, cnt = get_tile(new_adj_mat)
    plt.imshow(tile_mat)
    plt.show()
    print('count:', cnt)
    #check_correctness(adj_mat, new_adj_mat)
    
if opt.blockSZ_diff:
    adj_mat_path = '../data/{}/adj_mat.npy'.format(opt.dataset)
    adj_mat = np.where(get_normalized_adj(np.load(adj_mat_path)) > 0, 1, 0)
    get_blockSZ_diff_result(adj_mat, opt.dataset)
    
    
if opt.blockSZ_graph_diff:
    for dataset in ['METRLA', 'PEMSBAY', 'PEMS04', 'PEMS08']:
        adj_mat_path = '../data/{}/adj_mat.npy'.format(dataset)
        adj_mat = np.where(get_normalized_adj(np.load(adj_mat_path)) > 0, 1, 0)
        get_blockSZ_diff_result(adj_mat, dataset)
        
if opt.blockSZ_CORA_diff:
    opt.dataset = 'CORA'
    adj_mat_path = '../data/{}/adj_mat.npy'.format(opt.dataset)
    adj_mat = np.where(get_normalized_adj(np.load(adj_mat_path)) > 0, 1, 0)
    result, tiling_result = [], []
    new_adj_mat_list = []
    for num in [16, 32]:
        MAX = num
        new_adj_mat_list.append([reordering_by_simV2(adj_mat), reordering_by_simV3(adj_mat)])
        tiling_result.append(get_tiling_result(new_adj_mat_list[-1]))
        result.append([element[1] for element in tiling_result[-1]])
    idx = ['16x16', '32x32']
    cols = ['No reordering', 'simV2-jaccard', 'simV3-jaccard']
    df = pd.DataFrame(result, columns = cols, index = idx)
    df.to_csv('./graph_diff/CORA_result.csv')
    plt.figure(figsize=(20,20))
    for i in range(2):
        for j in range(1, 4):
            plt.subplot(2, 3, 3*i+j)
            if j == 1:
                plt.ylabel(idx[i], fontsize = 10)
            if i == 3:
                plt.xlabel(cols[j-1], fontsize = 9)
            #plt.imshow(np.array(tiling_result[i][j-1][0], dtype=float))
            plt.imshow(tiling_result[i][j-1][0])
    plt.tight_layout()
    plt.savefig('./graph_diff/CORA_result')
    plt.show()

if opt.toMCSR:
    adj_mat_path = '../data/{}/adj_mat.npy'.format(opt.dataset)
    adj_mat = get_normalized_adj(np.load(adj_mat_path)) 
    version = opt.version
    print('version: ', version)
    if version[0] == 'v':
        print('sim_func:', opt.sim_func)

    if version == 'origin':
        new_adj_mat = adj_mat
    elif version == 'degree':
        new_adj_mat = reordering_by_degree(adj_mat)
    elif version == 'v1':
        new_adj_mat = reordering_by_sim(adj_mat, opt.sim_func)
    elif version == 'v2':
        new_adj_mat = reordering_by_simV2(adj_mat, opt.sim_func)
    elif version == 'v3':
        new_adj_mat = reordering_by_simV3(adj_mat, opt.sim_func)
    elif version == 'v4':
        new_adj_mat = reordering_by_simV4(adj_mat, opt.sim_func)
    elif version == 'v5':
        new_adj_mat = reordering_by_simV5(adj_mat, opt.sim_func)
    check_correctness(adj_mat, new_adj_mat)
    CSR = sparse.csr_matrix(new_adj_mat)
    print('CSR:')
    print('row_ptr len:', len(CSR.indptr))
    print('col_idx len', len(CSR.indices))
    print('data_arr len:', len(CSR.data))
    print('total:', len(CSR.indptr) + len(CSR.indices) + len(CSR.data))
    MCSR = Modified_CSR(MAX)
    MCSR.transform(new_adj_mat)
    ret = MCSR.get_statistics()
    print('MCSR:')
    print('row_ptr len:', ret[0])
    print('col_idx len', ret[1])
    print('data_arr len:', ret[2])
    print('total:', sum(ret))

if opt.MCSR_compare:
    for dataset in ['METRLA', 'PEMSBAY', 'PEMS03', 'PEMS04', 'PEMS07', 'PEMS08']:
        adj_mat_path = '../data/{}/adj_mat.npy'.format(dataset)
        adj_mat = np.where(get_normalized_adj(np.load(adj_mat_path)) > 0, 1, 0)
        get_SIMD_len_diff(dataset, adj_mat)

if opt.proj_data:
    MAX = 2
    for dataset in ['METRLA', 'PEMSBAY', 'PEMS03', 'PEMS04', 'PEMS07', 'PEMS08']:
        print(dataset,':')
        adj_mat_path = '../data/{}/adj_mat.npy'.format(dataset)
        adj_mat = np.load(adj_mat_path)
        matrix = np.where(adj_mat > 0, 1, 0)
        s = '';
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                if i == matrix.shape[0] - 1 and j == matrix.shape[1] - 1:
                    s += str(matrix[i][j])
                else:
                    s += str(matrix[i][j]) + ', '
        path = "../data/{}/{}.txt".format(dataset, dataset)
        f = open(path, "w+")
        f.write(s)
        f.close()
        reordered_matrix = reordering_by_simV2(np.where(adj_mat > 0, 1, 0), opt.sim_func).astype(np.int32)
        s = '';
        for i in range(reordered_matrix.shape[0]):
            for j in range(reordered_matrix.shape[1]):
                if i == reordered_matrix.shape[0] - 1 and j == reordered_matrix.shape[1] - 1:
                    s += str(reordered_matrix[i][j])
                else:
                    s += str(reordered_matrix[i][j]) + ', '
        path = "../data/{}/{}_G.txt".format(dataset, dataset)
        f = open(path, "w+")
        f.write(s)
        f.close()
        CSR = sparse.csr_matrix(matrix)
        s = '';
        for i in range(len(CSR.indptr)):
            if i != len(CSR.indptr) - 1:
                s += str(CSR.indptr[i]) + ','
            else:
                s += str(CSR.indptr[i])
        path = "../data/{}/{}_csrindptr.txt".format(dataset, dataset)
        f = open(path, "w+")
        f.write(s)
        f.close()
        s = '';
        for i in range(len(CSR.indices)):
            if i != len(CSR.indices) - 1:
                s += str(CSR.indices[i]) + ','
            else:
                s += str(CSR.indices[i])
        path = "../data/{}/{}_csrindices.txt".format(dataset, dataset)
        f = open(path, "w+")
        f.write(s)
        f.close()
        s = '';
        for i in range(len(CSR.data)):
            if i != len(CSR.data) - 1:
                s += str(CSR.data[i]) + ','
            else:
                s += str(CSR.data[i])
        path = "../data/{}/{}_csrdata.txt".format(dataset, dataset)
        f = open(path, "w+")
        f.write(s)
        f.close()
        CSR = sparse.csr_matrix(reordered_matrix)
        s = '';
        for i in range(len(CSR.indptr)):
            if i != len(CSR.indptr) - 1:
                s += str(CSR.indptr[i]) + ','
            else:
                s += str(CSR.indptr[i])
        path = "../data/{}/{}_csrindptr_G.txt".format(dataset, dataset)
        f = open(path, "w+")
        f.write(s)
        f.close()
        s = '';
        for i in range(len(CSR.indices)):
            if i != len(CSR.indices) - 1:
                s += str(CSR.indices[i]) + ','
            else:
                s += str(CSR.indices[i])
        path = "../data/{}/{}_csrindices_G.txt".format(dataset, dataset)
        f = open(path, "w+")
        f.write(s)
        f.close()
        s = '';
        for i in range(len(CSR.data)):
            if i != len(CSR.data) - 1:
                s += str(CSR.data[i]) + ','
            else:
                s += str(CSR.data[i])
        path = "../data/{}/{}_csrdata_G.txt".format(dataset, dataset)
        f = open(path, "w+")
        f.write(s)
        f.close()
        print('row_ptrG len:', len(CSR.indptr))
        print('col_idxG len', len(CSR.indices))
        print('data_arrG len:', len(CSR.data))
        MCSR = Modified_CSR(2)  
        MCSR.transform(matrix)
        print(MCSR.get_statistics())
        s = '';
        for i in range(len(MCSR.row_ptr)):
            if i != len(MCSR.row_ptr) - 1:
                s += str(MCSR.row_ptr[i]) + ', '
            else:
                s += str(MCSR.row_ptr[i])
        path = "../data/{}/{}_rp.txt".format(dataset, dataset)
        f = open(path, "w+")
        f.write(s)
        f.close()
        s = '';
        for i in range(len(MCSR.col_idx)):
            if i != len(MCSR.col_idx) - 1:
                s += str(MCSR.col_idx[i]) + ', '
            else:
                s += str(MCSR.col_idx[i])
        path = "../data/{}/{}_ci.txt".format(dataset, dataset)
        f = open(path, "w+")
        f.write(s)
        f.close()
        s = '';
        for i in range(len(MCSR.data_arr)):
            if i != len(MCSR.data_arr) - 1:
                s += str(MCSR.data_arr[i]) + ', '
            else:
                s += str(MCSR.data_arr[i])
        path = "../data/{}/{}_da.txt".format(dataset, dataset)
        f = open(path, "w+")
        f.write(s)
        f.close()
        MCSR.transform(reordered_matrix)
        print(MCSR.get_statistics())
        s = '';
        for i in range(len(MCSR.row_ptr)):
            if i != len(MCSR.row_ptr) - 1:
                s += str(MCSR.row_ptr[i]) + ', '
            else:
                s += str(MCSR.row_ptr[i])
        path = "../data/{}/{}_G_rp.txt".format(dataset, dataset)
        f = open(path, "w+")
        f.write(s)
        f.close()
        s = '';
        for i in range(len(MCSR.col_idx)):
            if i != len(MCSR.col_idx) - 1:
                s += str(MCSR.col_idx[i]) + ', '
            else:
                s += str(MCSR.col_idx[i])
        path = "../data/{}/{}_G_ci.txt".format(dataset, dataset)
        f = open(path, "w+")
        f.write(s)
        f.close()
        s = '';
        for i in range(len(MCSR.data_arr)):
            if i != len(MCSR.data_arr) - 1:
                s += str(MCSR.data_arr[i]) + ', '
            else:
                s += str(MCSR.data_arr[i])
        path = "../data/{}/{}_G_da.txt".format(dataset, dataset)
        f = open(path, "w+")
        f.write(s)
        f.close()

if opt.check_graph:
    adj_mat_path = '../data/{}/adj_mat.npy'.format(opt.dataset)
    adj_mat = np.load(adj_mat_path)
    get_sparsity(adj_mat);