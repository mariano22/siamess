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

def get_stats(fns, get_x, n_sample=1000):
    if n_sample!='all':
        fns = random.sample(fns, min(len(fns),n_sample))
    t = torch.stack([IntToFloatTensor()(ToTensor()(Resize(460)(get_x(fn)))) for fn in fns])
    return t.mean(), t.std()

def plot_pred(interp,i, fn_info):
    fn_info_str = f'{interp.ds.fns[i//2]} {interp.ds.fns[interp.ds.draw_memo[i]]}' if fn_info  else ''
    title = f', pred = {interp.pred[i]}, prob = {interp.prob[i]}, loss = {interp.loss[i]:.4f}\n' + fn_info_str
    SiameseImage(interp.ds[i][0], interp.ds[i][1], interp.ds[i][2]).show(more_info = title)
    SiameseImage(interp.inputs[0][i]+1, interp.inputs[1][i]+1, interp.y[i]).show(more_info = title)

def top_loss(interp, fn_info=False, kind='all', limit = 50):
    """ kind in {'all', 'frr', 'far'} """
    c = 0
    for i in interp.ids:
        if kind=='all' or (kind=='far' and ~interp.y[i].bool()) or (kind=='frr' and interp.y[i].bool()):
            plot_pred(interp,i,fn_info)
            c+=1
            if c>=limit: break

def stats(pred,y):
    r = dict()
    # Accuracy
    r['acc'] = (pred==y).float().mean().item()
    # Error rate
    r['err'] = (pred!=y).float().mean().item()
    # Falsos rechazos
    r['frr'] = (y.bool() & (pred!=y)).float().mean().item()
    # Falsas aceptaciones
    r['far'] = (~y.bool() & (pred!=y)).float().mean().item()
    # Precision
    r['precision'] = ( (y.bool() & (pred==y)).float().sum() / y.bool().float().sum() ).item()
    print(f'Accuracy: {r["acc"]}')
    print(f'Error rate: {r["err"]}')
    print(f'False Aceptation Ratio (errores no detectados): {r["frr"]}')
    print(f'False Rejection Ratio (falsas alarmas): {r["far"]}')
    print(f'Precision: {r["precision"]}')
    print(f'Recall (1-FAR): {r["far"]}')
    return r
