from fastbook import *

class SiameseModelNN(Module):
    """ Architecture: distance( merge( encoder(x1), encoder(x2) ) )
        where enconder and distance are neural networks.
        - merge could be 'cat','diff'.
        - arch is the basic archtecture for the encoder (i.e. resnet18) """
    def __init__(self, arch):
        self.encoder = create_body(arch,n_in=1)
        self.head =create_head(512*4,1)

    def forward(self, x1, x2):
        ftrs = torch.cat([self.encoder(x1), self.encoder(x2)], dim=1)
        return self.head(ftrs).squeeze()

def siamese_splitter(model):
    return [ params(model.encoder), params(model.head) ]


class SiameseModelWithDistance(Module):
    """ Architecture: distance_fn( encoder(x1)  , encoder(x2) ).
        - enconder is a neural networks.
        - distance_fn is not trainable.
        - arch is the basic archtecture for the encoder (i.e. resnet18).
        - specify feature_dim to set the embedding dimension otherwise the output
        of the default cnn_learner would be used as fastai does.
        """
    def __init__(self, arch, distance_fn, feature_dim=None):
        self.distance_fn = distance_fn
        self.encoder = create_body(arch, n_in=1)
        if feature_dim is None:
            self.head = nn.Sequential(AdaptiveConcatPool2d(), Flatten())
        else:
            self.head = create_head(512*2, feature_dim)

    def forward(self, x1, x2):
        feat1 = self.head(self.encoder(x1))
        feat2 = self.head(self.encoder(x2))
        return self.distance_fn(feat1,feat2)


def test_model(m,dls):
    b = dls.one_batch()
    with torch.no_grad():
        print(f't1.shape = {b[0].shape} t2.shape = {b[1].shape}')
        print(f'target.shape = {b[2].shape}')
        print(f'out.shape = {m(b[0],b[1]).shape}')

def see_params(m): return [p.shape for p in params(m)]
