
import mxnet as mx
from mxnet import gluon, autograd
from mxnet.gluon.data import DataLoader
import mxboard
from mxnet import symbol

import model
import data_loader as loader

import numpy as np
import time

from sklearn.metrics import precision_score, recall_score

# feat_path, knn_graph_path, label_path
ctx = mx.gpu()
nepoch = 100
batch_size = 64
k_at_hop=[200,5]
data = loader.TrainDatasetFromNP(r'D:\Workspace\Projects\ChaosLab.old\build\x64\pick\feats.npy',
                                 r'D:\Workspace\Projects\ChaosLab.old\build\x64\pick\knn.npy',
                                 r'D:\Workspace\Projects\ChaosLab.old\build\x64\pick\labels.npy', k_at_hop)

data_loader = DataLoader(dataset=data, batch_size=batch_size, shuffle=True)

def accuracy(pred, label):
    pred = pred > 0.5
    acc = mx.nd.mean(pred == label)
    #pred = torch.argmax(pred, dim=1).long()
    #acc = torch.mean((pred == label).float())
    pred = pred.asnumpy()
    label = label.asnumpy()
    p = precision_score(label, pred)
    r = recall_score(label, pred)
    return p,r,acc 

def Train():
    N = k_at_hop[0] * (k_at_hop[1] + 1) + 1
    gcn = model.GCN()
    #mlp = model.MLP()

    gcn.collect_params().initialize(mx.init.Normal(0.01), ctx=ctx)
    #mlp.collect_params().initialize(mx.init.Normal(0.01), ctx=ctx)

    # hybridize
    gcn.hybridize()
    #mlp.hybridize()

    trainer_gcn = gluon.Trainer(gcn.collect_params(), 'adam', {'learning_rate': 0.001}) #, 'beta1': 0.5
    #trainer_mlp = gluon.Trainer(mlp.collect_params(), 'adam', {'learning_rate': 0.001}) #, 'beta1': 0.5

    crit = gluon.loss.SigmoidBinaryCrossEntropyLoss(from_sigmoid=True)

    tick = time.time()
    ts = time.localtime(tick)
    stamp = time.strftime('%Y%m%d%H%M%S', ts)
    with mxboard.SummaryWriter(logdir='logs/'+ stamp) as sw:
        iternum = 0
        for epoch in range(nepoch):
            for feat, A, center_idx, one_hop_idcs, edge_labels in data_loader:
                feat = feat.as_in_context(ctx)
                A = A.as_in_context(ctx)
                center_idx = center_idx.as_in_context(ctx)
                one_hop_idcs = one_hop_idcs.as_in_context(ctx)
                edge_labels = edge_labels.as_in_context(ctx)

                batch_size,_,_ = feat.shape

                w = mx.nd.zeros((batch_size, N),ctx=ctx)
                for b in range(batch_size):
                    w[b][one_hop_idcs[b]]=1

                with autograd.record():
                    #x = gcn(feat, A)
                    #_,_,dout = x.shape
                    #x = x.reshape(-1, dout)
                    pred = gcn(feat, A)
                    labels = edge_labels.reshape(-1, 1)
                    loss = crit(pred, labels).reshape((batch_size, -1)) * w
                loss.backward()

                trainer_gcn.step(batch_size)
                pred = pred.reshape(batch_size, -1)
                labels = labels.reshape(batch_size, -1)

                pred_ = mx.nd.zeros((batch_size, 200))
                labels_ = mx.nd.zeros((batch_size, 200))
                for b in range(batch_size):
                    pred_[b] = pred[b][one_hop_idcs[b]]
                    labels_[b] = labels[b][one_hop_idcs[b]]
                    pass

                lr = trainer_gcn.learning_rate
                p,r,acc = accuracy(pred_.reshape(-1), labels_.reshape(-1))

                sw.add_scalar(tag='Eva', value=('Acc', acc.mean().asscalar()), global_step=iternum)
                sw.add_scalar(tag='Eva', value=('P', p), global_step=iternum)
                sw.add_scalar(tag='Eva', value=('R', r), global_step=iternum)
                sw.add_scalar(tag='Loss', value=('loss', loss.mean().asscalar()), global_step=iternum)
                print("Loss:", loss.mean().asscalar(), "Acc:",acc.mean().asscalar(), "P:", p, "R:", r)
 
                if iternum % 2000 == 0:
                    #trainer_gcn.set_learning_rate(lr * 0.1)
                    gcn.export('./models/gcn-'+stamp, iternum / 2000)
                    #mlp.export('./models/mlp-'+stamp, iternum / 2000)
                    pass

                iternum = iternum + 1
                pass
            pass
        pass
    pass

if __name__ == '__main__':
    #net = model.GCN()
    #net.collect_params().initialize(mx.init.Normal(0.01), ctx=ctx)
    #x = mx.nd.ones((2, 5, 512), ctx=ctx)
    #A = mx.nd.ones((2, 5,5), ctx=ctx)
    #y = net(x, A)
    #print(y.shape)

    Train()
    pass
