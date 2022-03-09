from libraries_import_all import * 

from visualization import *
from dl_models import *
from dataset_siamese import *

@patch
def flatten(self:L): return L(x for xs in self for x in xs)

def empty_dir(dir_fp):
    print("Are you sure to delete {}? ['yes' for yes]".format(dir_fp))
    if input()=='yes':
        if os.path.isdir(dir_fp):
            shutil.rmtree(dir_fp)
        os.mkdir(dir_fp)
    else:
        assert False

def preprocess_choose(img):
    """ Choose between different preprocess methods. """
    npimg = np.array(img)
    _,otsu = cv2.threshold(npimg,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    _,otsugauss = cv2.threshold(cv2.GaussianBlur(npimg,(5,5),0),0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    _,fixth = cv2.threshold(npimg,127,255,cv2.THRESH_BINARY)
    adap = cv2.adaptiveThreshold(npimg,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,11,2)
    adapgauss = cv2.adaptiveThreshold(npimg,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,2)

    titles = ['Original Image', 'Global Thresholding (v = 127)',
              'Adaptive Mean Thresholding', 'Adaptive Gaussian Thresholding', 'Otsu', 'Otsu+Gaussian']
    images = [npimg, fixth, adap, adapgauss, otsu, otsugauss]
    plot_images_grid(images,width=2, titles=titles, figsize=(10,6))

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

def precision_score_recall_constrained(target, decision_values, required_recall=0.9):
    precision, recall, thresholds = sklearn.metrics.precision_recall_curve(target,decision_values)
    max_precision_by_recall = precision[recall>=required_recall].max()
    t = thresholds[recall[:-1]>required_recall].max()
    return max_precision_by_recall, t, (precision, recall, thresholds) 

def precision_recall_accuracy_f1_scores(p,y,th):
    pred = p >= th
    return precision_score(y,pred), recall_score(y, pred), accuracy_score(y, pred), f1_score(y, pred)

def calc_stats(pred,y,):
    if not isinstance(pred, torch.Tensor): pred = tensor(pred)
    if not isinstance(y, torch.Tensor): y = tensor(y)
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
    return r

def print_stats(r):
    print(f'Accuracy: {r["acc"]}')
    print(f'Error rate: {r["err"]}')
    print(f'False Aceptation Ratio (errores no detectados): {r["frr"]}')
    print(f'False Rejection Ratio (falsas alarmas): {r["far"]}')
    print(f'Precision: {r["precision"]}')
    print(f'Recall (1-FAR): {1-r["far"]}')
