from pprint import pprint

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from numpy import loadtxt
from scipy.signal import step2
from scipy.sparse import csr_matrix
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from sklearn.neighbors import NearestNeighbors, KDTree
import umap
import time
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import networkx as nx
import os



# os.environ["OMP_NUM_THREADS"] = "1"
from networkx.algorithms.community import asyn_lpa_communities
from sklearn.cluster import SpectralClustering

pathData = r"D:\letter.txt"
data = loadtxt(pathData)
label = data[:, -1]
dataset = data[:, :-1]

# umap
reducer = umap.UMAP(random_state=42)
dataset = reducer.fit_transform(dataset)


# knn
# def Weighted_KNN(data, k):
#     rows_count = len(data)
#     k = min(k, rows_count - 1)  
#     tree = KDTree(data, leaf_size=30)  
#     distances, indices = tree.query(data, k=k + 1)   
#     return distances[:, 1:], indices[:, 1:]  

def Weighted_KNN (data,k):
    
    rows_count = len(data)
    k = min(k, rows_count - 1)
   

 
    NearestNeighbors_infomation  = NearestNeighbors(n_neighbors=k).fit(data)
    
    distances, indices = NearestNeighbors_infomation.kneighbors()
    return distances, indices

# distances, indices = Weighted_KNN(dataset, k=3)
# print("distances = ", distances)
# print("indices = ", indices)

# calculate th RNN
def Weighted_RNN(data, k):
    
    RNN_distance = {}
    RNN_indices = {}
    Knn_distances, Knn_indices = Weighted_KNN(data, k)
    for i in range(len(data)):
        RNN_indices[i] = []
        RNN_distance[i] = []
   
    for id, dist, indi in zip(range(len(data)), Knn_distances, Knn_indices):
        for d, i in zip(dist, indi):
            RNN_distance[i].append(d)
            RNN_indices[i].append(id)
    RNN_distance_list = list(RNN_distance.values())
    RNN_indices_list = list(RNN_indices.values())
    return RNN_distance_list, RNN_indices_list


# r_distance_list, r_indices_list = Weighted_RNN(dataset, k=3)
# print("r_distances = ", r_distance_list)
# print("r_indices = ", r_indices_list)


def mutual_nearest_neighbors(data, k):
    BKNN_distance = []
    BKNN_indices = []
    knn_distance, knn_indices = Weighted_KNN(data, k)
    rnn_distance, rnn_indices = Weighted_RNN(data, k)
    for id, dist1, indi1, dist2, indi2 in zip(range(len(data)), knn_distance, knn_indices, rnn_distance, rnn_indices):
        BKNN_indices.append([])
        BKNN_distance.append([])
        for d1, i1 in zip(dist1, indi1):
            for d2, i2 in zip(dist2, indi2):
                if i1 == i2:
                    common_indice = i1
                    common_distance = d1
                    BKNN_indices[id].append(common_indice)
                    BKNN_distance[id].append(common_distance)
    return BKNN_indices, BKNN_distance


# BK_diatance, BK_indices = mutual_nearest_neighbors(dataset, 3)
# print("BK_distances = ", BK_diatance)
# print("BK_indices = ", BK_indices)


def peel_noise(data, k):
    bknn_indices, bknn_distances = mutual_nearest_neighbors(data, k)
    new_bknn_indice = []
    new_bknn_distance = []
    bknn_indices_index = []
    noise_set = []
    for i in range(len(data)):
       
        if bknn_indices[i]:
            bknn_indices_index.append(i)
            new_bknn_indice.append(bknn_indices[i])
            new_bknn_distance.append(bknn_distances[i])
        else:
            noise_set.append(i)

    return bknn_indices_index, new_bknn_indice, new_bknn_distance, noise_set


def cosine_similarity_func(a, b):
 
    dot_product = np.dot(a, b)  
    norm_a = np.linalg.norm(a) 
    norm_b = np.linalg.norm(b)  

    if norm_a == 0 or norm_b == 0:
        return 0  
    else:
        return dot_product / (norm_a * norm_b)

# 计算局部密度值
def local_peel_values(data, k):
    new_mnn_indice, new_mnn_distance = mutual_nearest_neighbors(data, k)
    # print(new_mnn_indice)

    # weight = weighted_point(data, k)
    l = len(data)
    # element_sum = np.zeros(l)
    bordel_value = np.zeros(l)
    # factor = Weighting_factor(data)

    knn_distance, knn_indices = Weighted_KNN(data, k)

 
    # for i in range(len(data)):
    #     sum = np.sum(data[i]) / data.shape[version_1]
 
    #     if sum == factor:
    #         element_sum[i] = sum
    #     else:
    #         factor_pro = round(sum / factor, 2)

    #         if np.isnan(factor_pro) or np.isinf(factor_pro):
      
    #         element_sum[i] = factor_pro


    for id, mnn_indi, mnn_dist in zip(range(len(data)), new_mnn_indice, new_mnn_distance):
        # element_factor = element_sum[id]

        
        if not mnn_indi:
            bordel_value[id] = -1
            continue
        else:
        
            for indi, dist in zip(mnn_indi, mnn_dist):
                # print(indi)

                cosine_similarity = cosine_similarity_func(data[id], data[indi])

         
                bordel_value[id] += cosine_similarity
        bordel_value[id] = bordel_value[id] / len(mnn_indi)


        if np.isnan(bordel_value[id]) or np.isinf(bordel_value[id]):
            bordel_value[id] = 0  

    return bordel_value
# value = local_peel_values(dataset,3)
# print(value)


def outside_peel_values(data, k):
    new_mnn_indice, new_mnn_distance = mutual_nearest_neighbors(data, k)
    l = len(data)
    # element_sum = np.zeros(l)
    outside_border_value = np.zeros(l)

    border_value = local_peel_values(data, k)
    for id,neighbor in zip(range(len(data)), new_mnn_indice):
        # print(neighbor)
        if border_value[id] == -1:
            outside_border_value[id] = -1
            continue
        else :
            outside_border_value[id] = 0
            for indi in neighbor:
                # print(indi)
                outside_border_value[id] += border_value[indi]
        outside_border_value[id] = outside_border_value[id] / len(neighbor)
    return outside_border_value

# ouside = outside_peel_values(dataset,3)
# print(ouside)



def border_peel_single(data, k):

    local_border_value = local_peel_values(data, k)
    outside_border_value = outside_peel_values(data, k)

    mnn_index, mnn_indices, mnn_distance, noise_set = peel_noise(data, k)


    new_local_border_value = local_border_value[mnn_index]
    new_outside_border_value = outside_border_value[mnn_index]
    l = len(new_local_border_value)

    density = np.zeros(l)

    for i in range(l):
        density[i] = local_border_value[i] - new_outside_border_value[i]
    # print(density)

    filter = density > 0
    return filter


# f = border_peel_single(dataset,3)
# print(f)

def border_peel(data, k):

    mnn_index, mnn_indices, mnn_distances, noise_set = peel_noise(data, k)

    mnn_index = np.array(mnn_index)

    noise = np.array(noise_set)

    origin_data = data[mnn_index]

    # border_values = peel_values(origin_data, k)
    # threshold = global_threshold(origin_data, k)

    filter = border_peel_single(data, k)
    core_points = origin_data[filter]
    border_points = origin_data[filter == False]
    core__index = mnn_index[filter]
    border__idnex = mnn_index[filter == False]

    # cluster_uf = union_find.UF(data_length)
    return core_points, border_points


def plot_core_boundary(core_points, border_points, data):
    plt.figure(figsize=(10, 8), dpi=300)
    '''
    plt.scatter(data[:, 0], data[:, version_1], c='red', label='noise_points', alpha=version_1,s=40)
    plt.scatter(core_points[:, 0], core_points[:, version_1], c='blue', label='core_points')
    plt.scatter(border_points[:, 0], border_points[:, version_1], c='black', label='border_points')
    plt.title("the division of data_points")
    plt.legend()
    plt.show()
    '''
    # plt.figure(figsize=(10, 8), dpi=600)

    is_3d = data.shape[1] == 3
    if is_3d:
        ax = plt.axes(projection='3d') 
        ax.scatter(data[:, 0], data[:, 1], data[:, 2], c='red', label='noise_points', alpha=0.8, s=40)
        ax.scatter(core_points[:, 0], core_points[:, 1], core_points[:, 2], c='blue', label='core_points')
        ax.scatter(border_points[:, 0], border_points[:, 1], border_points[:, 2], c='orange', label='border_points')

        ax.set_xlabel('X Axis')
        ax.set_ylabel('Y Axis')
        ax.set_zlabel('Z Axis')
        ax.set_title("The Division of Data Points ")
    else:
        plt.scatter(data[:, 0], data[:, 1], c='red', alpha=1, s=40)
        plt.scatter(core_points[:, 0], core_points[:, 1], c='blue')
        plt.scatter(border_points[:, 0], border_points[:, 1], c='cyan')
        plt.title("The Division of Data Points ")

    plt.legend()
    plt.savefig("t8-k_divide.eps", format='eps')
    plt.show()



def plot_separate_core_and_boundary(core_points, border_points):
    
    plt.figure(figsize=(10, 8), dpi=300)
    if len(core_points) > 0:
        plt.scatter(core_points[:, 0], core_points[:, 1], c='blue', label='core_points')
    # plt.title("show core", fontsize=16)
    plt.xlabel("X", fontsize=12)
    plt.ylabel("Y", fontsize=12)
    plt.legend(fontsize=10)
    # plt.grid(True)
    plt.show()

    
    plt.figure(figsize=(10, 8), dpi=600)
    if len(border_points) > 0:
        plt.scatter(border_points[:, 0], border_points[:, 1], c='orange', label='border_points')
    # plt.title("show border", fontsize=16)
    plt.xlabel("X", fontsize=12)
    plt.ylabel("Y", fontsize=12)
    plt.legend(fontsize=10)
    # plt.grid(True)
    plt.show()


# a,b,c,d = border_peel(dataset, k=28)
# # print(c)
# # print(d)
# core_point = dataset_copy[c]
# border_point = dataset_copy[d]
# plot_core_boundary(core_point, border_point, dataset_copy)
# plot_separate_core_and_boundary(core_point, dataset_copy)

def cluster_core_points(data, k,diversity_threshold = 0, overlap_threshold = 0):
    #step 1 : preparation

    core_points, border_points = border_peel(data, k)
   
    border_values = local_peel_values(data, k)
    # print(border_values)

    bknn_index, bknn_indices, bknn_distances, noise_set = peel_noise(data, k)
    bknn_index = np.array(bknn_index)


    core_border_values = border_values[bknn_index]
    filter = border_peel_single(data, k)

    core_index = bknn_index[filter]
  
    '''
    core_border_values = core_border_values[filter]

    sorted_idx = np.argsort(-core_border_values)
 
    sorted_core_values = core_border_values[sorted_idx]
    sorted_core_index = core_index[sorted_idx]
    '''

    core_distances,core_indices = Weighted_KNN(core_points,k)

    n = core_points.shape[0]
    rows, cols, data = [], [], []
    for i in range(n):
        for j in core_indices[i]:
            rows.append(i)
            cols.append(j)
            data.append(1)

    adj_matrix = csr_matrix((data, (rows, cols)), shape=(n, n))
    # print(adj_matrix)


    n_components, labels = connected_components(csgraph=adj_matrix, directed=False, return_labels=True)
    # print(labels)
    return labels, adj_matrix,core_index

# w ,r,q= cluster_core_points(dataset,9)
# print(w)
# print(q)
def cluster_border_points(data, k):

    labels, matrix,core_index = cluster_core_points(data, k)
    clusters = np.ones(len(data)) * -1
    for i in range(len(labels)):
        clusters[core_index[i]] = labels[i]

    core_points, border_points = border_peel(data, k)

    mnn_index, mnn_indices, mnn_distances, noise_set = peel_noise(data, k)
    print(len(noise_set))
    mnn_index = np.array(mnn_index)
    filter = border_peel_single(data, k)

    border_index = mnn_index[filter == False]

    for i, indi, dist in zip(mnn_index, mnn_indices, mnn_distances):
  
        for index in border_index:
          
            if i == index:
               
                mnn = np.array(indi)
                
                mnn_d = np.array(dist)
             
                mnn_d_index = np.argsort(mnn_d)[::-1]
                for j in range(len(mnn_d_index)):
                    if clusters[mnn[mnn_d_index[j]]] != -1:
                        clusters[index] = clusters[mnn[mnn_d_index[j]]]
                        break
                    else:
                        continue


    return clusters
# v = cluster_border_points(dataset,6)
# print(v)
# ari = adjusted_rand_score(label,v)
# nmi = normalized_mutual_info_score(label,v)
# print(ari, nmi)
"""

def cluster_noise(data, k, m):
    mnn_index, mnn_indices, mnn_distances, noise_set = peel_noise(data, k)
    clusters_core, centroids = cluster_core_points(data, k, m)
    clusters = cluster_border_points(data, k, m)
    noise_set = np.array(noise_set)
    for i in noise_set:
        noise = data[i]
        distances = []
        for cent in centroids:
            # for cent in centroids:
                # dot_product = np.dot(data[i], cent)  
                # norm_a = np.linalg.norm(data[i])  
                # norm_b = np.linalg.norm(cent)
                # similarity = norm_a * norm_b
                dist = np.linalg.norm(noise - cent)
                distances.append(dist)
                b_index = np.argmin(distances)
                clusters[i] = b_index

    return clusters
"""


def plot_clusters(data, clusters):

    valid_indices = clusters != -1  
    valid_data = data[valid_indices]  
    valid_clusters = clusters[valid_indices]  


    plt.figure(figsize=(10, 8), dpi=600)

    if valid_data.shape[1] == 3:  
        ax = plt.axes(projection='3d')
        ax.scatter(valid_data[:, 0], valid_data[:, 1], valid_data[:, 2], c=valid_clusters, cmap='tab10', s=30,
                   alpha=0.8)
        ax.set_xlabel("X", fontsize=12)
        ax.set_ylabel("Y", fontsize=12)
        ax.set_zlabel("Z", fontsize=12)
    else:

        plt.scatter(valid_data[:, 0], valid_data[:, 1], c=valid_clusters, cmap='tab10', s=30, alpha=0.8)
        plt.xlabel("X", fontsize=12)
        plt.ylabel("Y", fontsize=12)

    # plt.title("final_clustering", fontsize=16)
    plt.savefig(r"D:\WNIBP\mwbp\smile2.eps", format='eps')
    # plt.colorbar(label='CBH')
    # plt.show()
    plt.show()




 #plot_clusters(dataset_copy, clusters)

# clusters= cluster_border_points(dataset, k=28)
# corep = dataset_copy[]

# plot_clusters(dataset_copy, clusters)

ARI = []
NMI = []
index_list = []
results = []
best_ari = 0
best_nmi = 0
best_v = None
best_g = None
# for v in range(2,15):
# k = 4
for k in range(3,20):
    #     g = 3
    print(f"Processing k={k}")
            # start_time = time.time()
    u = cluster_border_points(dataset,k)
    ari = adjusted_rand_score(label, u)
    nmi = normalized_mutual_info_score(label, u)
    print(f"ARI: {ari}, NMI: {nmi}")
# plot_clusters(dataset_copy, u)
        # Append scores and parameters
    # ARI.append(ari)
    # NMI.append(nmi)
    # index_list.append(k)

        # Check if this is the best ARI/NMI
    # if ari > best_ari:
    #     best_ari = ari
    #     best_k = k
    # if nmi > best_nmi:
    #     best_nmi = nmi
    #     best_k = k
    # results.append({"k": k, "ARI": ari, "NMI": nmi})

# 转换为 DataFrame
# df_results = pd.DataFrame(results)
# output_path = r"D:\WNIBP\version_2\T8txt_Results.xlsx"
#
# # 保存到 Excel
# df_results.to_excel(output_path, index=False)
# print(f"结果已保存到 {output_path}")
# # Output results
# print("Max ARI:", best_ari, "at v =", best_v, "g =", best_g)
# print("Max NMI:", best_nmi, "at v =", best_nmi_v, "g =", best_nmi_g)

# start_time = time.time()
# u =  cluster_noise(dataset,8,4)
# end_time = time.time()
# execution_time = end_time - start_time
# print(f"代码执行时间: {execution_time} 秒")
# print(u)
