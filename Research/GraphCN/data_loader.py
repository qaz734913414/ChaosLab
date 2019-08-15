
import mxnet as mx
from mxnet import gluon
from mxnet.gluon.data import dataset
from mxnet.gluon.data.vision.transforms import ToTensor, Compose, Resize, Normalize, Cast
from mxnet.gluon.data import DataLoader

import numpy as np



class TrainDatasetFromNP(dataset.Dataset):
    def __init__(self, feat_path, knn_graph_path, label_path, 
                 k_at_hop=[200,5], active_connection=5, is_train = True):
        super(TrainDatasetFromNP, self).__init__()
        self.features = np.load(feat_path)
        self.knn_graph = np.load(knn_graph_path)[:,:k_at_hop[0]+1]
        self.labels = np.load(label_path).astype('int64').reshape(-1)
        self.num_samples = len(self.features)
        self.depth = len(k_at_hop)
        self.k_at_hop = k_at_hop
        self.active_connection = active_connection
        self.is_train = is_train
        pass


    def __len__(self):
        return self.num_samples

    def __getitem__(self, index):
        '''
        return the vertex feature and the adjacent matrix A, together 
        with the indices of the center node and its 1-hop nodes
        '''
        # hops[0] for 1-hop neighbors, hops[1] for 2-hop neighbors
        hops = list()
        center_node = index 
        hops.append(set(self.knn_graph[center_node][1:]))

        # Actually we dont need the loop since the depth is fixed here,
        # But we still remain the code for further revision
        for d in range(1,self.depth): 
            hops.append(set())
            for h in hops[-2]:
                hops[-1].update(set(self.knn_graph[h][1:self.k_at_hop[d]+1]))

        hops_set = set([h for hop in hops for h in hop])
        hops_set.update([center_node,])
        unique_nodes_list = list(hops_set) 
        unique_nodes_map = {j:i for i,j in enumerate(unique_nodes_list)}

        center_idx = mx.nd.array([unique_nodes_map[center_node],])
        one_hop_idcs = mx.nd.array([unique_nodes_map[i] for i in hops[0]])
        #hop_idcs = mx.nd.array([unique_nodes_map[i] for hop in hops for i in hop])

        center_feat = mx.nd.array(self.features[center_node])
        feat = mx.nd.array(self.features[unique_nodes_list])
        feat = feat - center_feat

        max_num_nodes = self.k_at_hop[0] * (self.k_at_hop[1] + 1) + 1
        num_nodes = len(unique_nodes_list)
        A = mx.nd.zeros((num_nodes, num_nodes))

        _, fdim = feat.shape
        if max_num_nodes > num_nodes: # fault tolerance
            feat = mx.nd.concat(feat, mx.nd.zeros((max_num_nodes - num_nodes, fdim)), dim=0)
            pass
      
        for node in unique_nodes_list:
            neighbors = self.knn_graph[node, 1:self.active_connection+1]
            for n in neighbors:
                if n in unique_nodes_list: 
                    A[unique_nodes_map[node], unique_nodes_map[n]] = 1
                    A[unique_nodes_map[n], unique_nodes_map[node]] = 1

        D = A.sum(axis=1)
        print(D)
        A = mx.nd.broadcast_div(A, D)
        A_ = mx.nd.zeros((max_num_nodes,max_num_nodes))
        A_[:num_nodes,:num_nodes] = A

        labels = self.labels[np.asarray(unique_nodes_list)]
        labels = mx.nd.array(labels)

        labels_ = mx.nd.zeros((max_num_nodes)) - 1
        labels_[:num_nodes] = labels

        center_label = labels[center_idx]
        edge_labels = (center_label == labels_)

        if self.is_train:
            return feat, A_, center_idx, one_hop_idcs, edge_labels

        # Testing
        one_hop_labels = labels[one_hop_idcs]
        #center_label = labels[center_idx]
        edge_labels = (center_label == one_hop_labels)

        unique_nodes_list = mx.nd.array(unique_nodes_list)
        unique_nodes_list = mx.nd.concat(unique_nodes_list, mx.nd.zeros((max_num_nodes-num_nodes)), dim=0)

        return feat, A_, center_idx, one_hop_idcs, unique_nodes_list, edge_labels


if __name__ == '__main__':
    # To test data loader
    k_at_hop=[200,5]
    batch_size = 1
    data = TrainDatasetFromNP(r'E:\vggface2_train\new\itest\feats.npy',
                              r'E:\vggface2_train\new\itest\neighbors.npy',
                              r'E:\vggface2_train\new\itest\labels.npy', k_at_hop)

    data_loader = DataLoader(dataset=data, batch_size=batch_size, shuffle=False)
    for feat, A, center_idx, one_hop_idcs, edge_labels in data_loader:
        print(feat)
        print(A)
        #np.save('feat.npy', feat.asnumpy())
        #np.save('A.npy', A.asnumpy())
        #np.save('ohi.npy', one_hop_idcs.asnumpy())
        #np.save('el.npy', edge_labels.asnumpy())
        break

    pass