from fastbook import *

class SiameseModelNN(Module):
    """ Architecture: distance( merge( encoder(x1), encoder(x2) ) )
        where enconder and distance are neural networks.
        - merge could be 'cat','diff'.
        - arch is the basic archtecture for the encoder (i.e. resnet18) """
    def __init__(self, arch):
        self.encoder = create_body(arch,n_in=3)
        self.head =create_head(512*4,1)

    def splitter(model):
        return [ params(model.encoder), params(model.head) ]

    def forward(self, x1, x2):
        ftrs = torch.cat([self.encoder(x1), self.encoder(x2)], dim=1)
        return self.head(ftrs).squeeze()


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
        self.encoder_body = create_body(arch, n_in=3)
        if feature_dim is None:
            self.encoder_head = nn.Sequential(AdaptiveConcatPool2d(), Flatten())
        else:
            self.encoder_head = create_head(512*4, feature_dim)
        self.encoder = nn.Sequential(self.encoder_body, self.encoder_head)

    def forward(self, x1, x2):
        feat1 = self.encoder(x1)
        feat2 = self.encoder(x2)
        return self.distance_fn(feat1,feat2)

    def splitter(model):
        return [ params(model.encoder_body), params(model.encoder_head) ]


def test_model(m,dls,withdistance=False):
    b = dls.one_batch()
    with torch.no_grad():
        print(f't1.shape = {b[0].shape} t2.shape = {b[1].shape}')
        e0 = m.encoder(b[0])
        e1 = m.encoder(b[1])
        print(f'encoder(t1).shape = {e0.shape} encoder(t2).shape = {e1.shape}')
        if hasattr(m, 'distance_fn'):
            emb0 = m.encoder_body(b[0])
            emb1 = m.encoder_body(b[1])
            print(f'encoder_body(t1).shape = {emb0.shape} encoder_body(t2).shape = {emb1.shape}')
        print(f'target.shape = {b[2].shape}')
        out = m(b[0],b[1])
        print(f'out.shape = {out.shape}')

def see_params(m): return [p.shape for p in params(m)]

def my_loss_func_trivial(dists, target):
    return torch.where(target.bool(), dists, -dists).sum()

def my_loss_func_LeCun(dist, target, margin=2, reduction='mean'):
    assert reduction in ['mean', 'none']
    neg_dist = torch.clamp(margin - dist, min=0.0)
    res = torch.where(target.bool(), dist, -neg_dist).pow(2)
    if reduction=='mean':
        res = res.mean()
    return 0.5 * res

def accuracy_dist(inp, targ, thresh=1):
    inp,targ = flatten_check(inp,targ)
    return ((inp<thresh)==targ.bool()).float().mean()

def thresh_finder(preds, targs, acc, x0, xf):
    xs = torch.linspace(x0,xf)
    accs = [ acc(preds, targs, thresh=x) for x in xs ]
    plt.plot(xs,accs)

class Interpret:
    def __init__(self, learn):
        self.learn = learn
        if isinstance(learn.model, SiameseModelWithDistance):
            self.model_type = 'dist'
        elif isinstance(learn.model, SiameseModelNN):
            self.model_type = 'nn'
        else:
            assert(False)

    def set_dl(self, dl, ds, d_thresh = 1):
        self.ds, self.ds = dl, ds
        self.inputs, self.act, self.y, self.loss = self.learn.get_preds(dl=dl, with_loss=True, with_input=True)
        if self.model_type == 'nn':
            self.prob = self.act.sigmoid()
            self.pred = self.prob > 0.5
        elif self.model_type == 'dist':
            self.pred = self.act < d_thresh
        else:
            assert(False)
        self.ids = torch.argsort(self.loss, descending=True)

    def debug(self,idx):
        assert(interp.ds[idx][2] == interp.y[idx])
        print(f'y = {interp.y[idx]}')
        print(f'pred = {interp.pred[idx]}')
        print(f'loss = {interp.loss[idx]}')
        print(f'act = {interp.act[idx]}')
        print(f'prob = {interp.prob[idx]}')
        display(interp.ds[idx][0])
        display(interp.ds[idx][1])
        show_image(self.inputs[0][idx]+1)
        show_image(self.inputs[1][idx]+1)
