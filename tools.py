from fastbook import *
from dl_models import *
from dataset_siamese import *

def plotpca2d(xs_pca, ys, cls_start=None, n_cls=None):
    assert (cls_start==None) == (n_cls==None)
    if cls_start is None:
        cls_start = 0
        n_cls = len(set(ys.tolist()))
    fig = plt.figure()
    plt.title('Embeddings')

    cm = plt.get_cmap('gist_rainbow')
    ax =  plt.axes()
    ax.set_prop_cycle('color', [cm(1.*i/n_cls) for i in range(n_cls)])

    for i in range(cls_start,cls_start+n_cls):
        plt.plot(xs_pca[ys==i,0], xs_pca[ys==i,1], '.', label=f'class {i}')

def plotpca3d(xs_pca, ys, cls_start=None, n_cls=None):
    assert (cls_start==None) == (n_cls==None)
    if cls_start is None:
        cls_start = 0
        n_cls = len(set(ys.tolist()))
    fig = plt.figure(figsize=(10,10))
    plt.title('Embeddings')

    cm = plt.get_cmap('gist_rainbow')
    ax = plt.axes(projection='3d')
    ax.set_prop_cycle('color', [cm(1.*i/n_cls) for i in range(n_cls)])

    for i in range(cls_start,cls_start+n_cls):
        ax.plot3D(xs_pca[ys==i,0], xs_pca[ys==i,1], xs_pca[ys==i,2], '.', label=f'class {i}')

def get_stats(fns, get_tensor, n_sample=1000):
    if n_sample!='all':
        fns = random.sample(fns, min(len(fns),n_sample))
    t = torch.stack([get_tensor(fn) for fn in fns])
    return t.mean(), t.std()

def top_loss(tlids, pred, y, loss, ds, fn_info=False, kind='all', limit = 50):
    """ kind in {'all', 'frr', 'far'} """
    c = 0
    for i in tlids:
        if kind=='all' or (kind=='far' and ~y[i].bool()) or (kind=='frr' and y[i].bool()):
            fn_info_str = f'{ds.fns[i//2]} {ds.fns[ds.draw_memo[i]]}' if fn_info  else ''
            show_siamese(ds[i][0], ds[i][1], ds[i][2],
                         more_info = f'pred:{pred[i]} loss:{loss[i]:.4f}' + fn_info_str,
                         cmap='binary', figsize=(5,5))
            c+=1
            if c>=limit: break

# Inspeccionando la validacion
class Interprete:
    def __init__(self, learn):
        self.learn = learn
        if isinstance(learn.model, SiameseModelWithDistance):
            self.model_type = 'dist'
        elif isinstance(learn.model, SiameseModelNN):
            self.model_type = 'mm'
        else:
            assert(False)

    def set_dl(self, dl, ds, d_thresh = 1):
        self.ds, self.ds = dl, ds
        self.act, self.y, self.loss = self.learn.get_preds(dl=dl, with_loss=True)
        if self.tmodel == 'nn':
            self.pred = self.act.sigmoid() > 0.5
        else:
            self.pred = self.act < d_thresh
        self.ids = torch.argsort(self.loss, descending=True)

        # Accuracy
        self.acc = (self.pred==self.y).float().mean().item()
        # Error rate
        self.err = (self.pred!=self.y).float().mean().item()
        # Falsos rechazos
        self.frr = (self.y.bool() & (self.pred!=self.y)).float().mean().item()
        # Falsas aceptaciones
        self.far = (~self.y.bool() & (self.pred!=self.y)).float().mean().item()

    def stats(self):
        print(f'Accuracy: {self.acc}')
        print(f'Error rate: {self.err}')
        print(f'False Aceptation Ratio: {self.frr}')
        print(f'False Rejection Ratio: {self.far}')
