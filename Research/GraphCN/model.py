
import mxnet as mx
from mxnet import gluon
from mxnet.gluon import nn

# input BxNxD
class GraphConv(gluon.HybridBlock):
    def __init__(self, in_channels, channels):
        super(GraphConv, self).__init__()
        with self.name_scope():
            #self.conv = nn.Dense(units=channels, in_units=2*in_channels, activation='relu') # 这个地方是不对的
            self.weight = self.params.get('weight',shape=(channels, 2*in_channels), init='xavier', dtype='float32', allow_deferred_init=True)
            self.bias = self.params.get('bias', shape=channels, init='zeros', allow_deferred_init=True)
            self.relu = nn.Activation('relu')
        pass

    def hybrid_forward(self, F, x, A, weight, bias):
        f = F.concat(x, F.batch_dot(A, x), dim=2)
        y = F.FullyConnected(data=f, weight=weight, bias=bias, num_hidden=self.weight.shape[0], flatten=False, no_bias=False) # BNDxDF=BNF
        z = self.relu(y)
        return z
    pass


class GCN(gluon.HybridBlock):
    def __init__(self):
        super(GCN, self).__init__()
        with self.name_scope():
            self.bn = nn.BatchNorm(in_channels=512, axis=2)
            self.conv1 = GraphConv(512,512)
            self.conv2 = GraphConv(512,512)
            self.conv3 = GraphConv(512,256)
            self.conv4 = GraphConv(256,256)

            self.conv5 = nn.Dense(256)
            self.prelu = nn.PReLU()
            self.conv6 = nn.Dense(1, activation='sigmoid')
        pass

    def hybrid_forward(self, F, x, A):
        # BND
        x = self.bn(x)
        x = self.conv1(x, A)
        x = self.conv2(x, A)
        x = self.conv3(x, A)
        x = self.conv4(x, A)

        x = F.reshape(x, (-3, 0))

        x = self.conv5(x)
        x = self.prelu(x)
        x = self.conv6(x)
        return x
    pass

