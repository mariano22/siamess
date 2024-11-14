from libraries_import_all import *

from features_grid import *
from grid_search import *
from methodology import *

def otsu_preprocess(img):
    """ Apply the choosen preprocessing to a PIL Image """
    _,np_img = cv2.threshold(cv2.GaussianBlur(np.array(img),(5,5),0),0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    np_img = 255-np_img
    return Image.fromarray(np_img)

def preproc_pipeline(path, get_x, f):
    nf = Path(path)/Path(f).name
    img = otsu_preprocess(get_x(f)).save(nf)
    return nf

def data_augmentation(img, n_augmentation):
    """ np_img: numpy.array of image to apply n_augmentation random data augmentation.
        Selected parameters are choosen to make subtle change on signatures images. """
    transformation=iaa.Affine(rotate=(-15,15),scale={"x": (0.75, 1.25), "y": (0.75, 1.25)})
    return [  Image.fromarray(transformation.augment_image(np.array(img))) for i in range(n_augmentation) ]


def preproc_pipeline(out_path, get_x, f):
    nf = Path(out_path)/Path(f).name
    img = otsu_preprocess(get_x(f)).save(nf)
    return nf

def feature_extract_pipeline(out_path, get_x, ndiv, f):
    nf = str(Path(out_path)/Path(f).name)[:-3]+'pkl'
    FastGrid(f, get_x, N_divisions=ndiv).save(nf)
    return nf

def load_features(ndiv):
    FEATURES_PATH = './data/parodi_out/features/'+str(ndiv)
    fns = L()
    xs = L()
    for npy_fp in glob.glob(FEATURES_PATH+'/*.npy'):
        with open(npy_fp, 'rb') as f:
            xs.append(np.load(f))
        with open(npy_fp[:-3]+'txt', 'r') as f:
            fns.extend( L(f.readlines()).map(lambda x : x.rstrip()) )
    x = np.concatenate(xs)
    return fns,x

class GridFeatureGetter:
    def __init__(self, ndiv):
        self.fns,self.x=load_features(ndiv)
        self.fns=self.fns.map(lambda x : Path(x).name)
    def __call__(self,f): 
        if isinstance(f,str): f=Path(f)
        return self.x[self.fns.index(f.name)]

def get_decision_values(clf, x):
    if hasattr(clf, "predict_proba"):
        y_proba = clf.predict_proba(x)
        assert len(y_proba.shape)==2 and y_proba.shape[1]==2
        assert len(clf.classes_)==2 and (clf.classes_==[False,True]).all()
        y_proba = y_proba[:,1]
    else:
        y_proba = clf.decision_function(x)
        len(y_proba.shape)==1
    return y_proba

def normalize(scalers, features):
    for i in range(features.shape[1]):
        features[:,i,:]=scalers[i].transform(features[:,i,:])
    return features

def featurize_pairs(pairs, features):
    x1, x2, f = tuple(map(L,zip(*pairs)))
    X1, X2, Y = features[x1], features[x2], np.array(f)
    X = np.stack([X1,X2],axis=1)
    X = X1-X2
    X=X.reshape((X.shape[0],-1))
    return X,Y,X1, X2

def augment_x(x,K): return L(Path(str(x.name)[:-7] + str(i) + '_6g.PNG') for i in range(min(10,K)))
def augment_y(y,K): return L(y for i in range(min(10,K)))

PARODI_GRID_SEARCH_OPTIONS = {
    'FEATURE_MIX_METHOD' : ['FeatureObtainer_Concat', 'FeatureObtainer_Difference'],
    'FFT' : ['OFF', 'ON'],
    'N_DIVISIONS' : ['8','16','32','64', '128'],
    'DATASET_AUGMENTATION' : ['OFF', 'ON'],
    'SCALER_METHOD' : ['NONE', 'StandardScaler', 'QuantileTransformer', 'MinMaxScaler', 'Normalizer'],
}

RF_GRID_SEARCH_OPTIONS = {
    'FEATURE_MIX_METHOD' : ['FeatureObtainer_Concat', 'FeatureObtainer_Difference'],
    'FFT' : ['OFF', 'ON'],
    'N_DIVISIONS' : ['8','16','32','64', '128'],
    'DATASET_AUGMENTATION' : ['OFF', 'ON'],
    'SCALER_METHOD' : ['NONE'],
}

def split_model_threshold_train(all_train, threshold_train_cls_set):
    all_train_cls_set = set(all_train[1])

    model_train_cls_set = all_train_cls_set.difference(threshold_train_cls_set)

    is_model_train = all_train[1].map(lambda cls : cls in model_train_cls_set)
    model_train = all_train[0][is_model_train], all_train[1][is_model_train]
    threshold_train = all_train[0][~is_model_train], all_train[1][~is_model_train]

    print(f'# train signatures: {len(all_train[0])} | # train model signatures {len(model_train[0])} | # train threshold signatures {len(threshold_train[0])} | check {len(model_train[0])+len(threshold_train[0])}')
    return model_train,threshold_train

def augment_signatures(ds,K):
    return ds[0].map(partial(augment_x,K=K)).flatten(), ds[1].map(partial(augment_y,K=K)).flatten()

def parodi_configurate(config, all_train, threshold_train_cls_set,  valid, valid_pairs, augment_ratio=4):
    global PARODI_GRID_SEARCH_OPTIONS
    assert config_is_well_typed(config, PARODI_GRID_SEARCH_OPTIONS)
    fn2x = GridFeatureGetter(int(config['N_DIVISIONS']))

    model_train, threshold_train = split_model_threshold_train(all_train, threshold_train_cls_set)
    
    if config['DATASET_AUGMENTATION']=='ON':
        model_train = augment_signatures(model_train, augment_ratio)
    
    model_train_features = np.array(model_train[0].map(fn2x))
    threshold_train_features = np.array(threshold_train[0].map(fn2x))
    valid_features = np.array(valid[0].map(fn2x))
    
    if config['SCALER_METHOD']!='NONE':
        scalers = [ eval(config['SCALER_METHOD'])() for _ in range(model_train_features.shape[1]) ]
        for i in range(model_train_features.shape[1]):
            scalers[i].fit(np.array(L(model_train[0]+threshold_train[0]).map(fn2x))[:,i,:])
        model_train_features = normalize(scalers,model_train_features)
        threshold_train_features = normalize(scalers,threshold_train_features)
        valid_features = normalize(scalers,valid_features)
    
    model_ds_pairs = make_pairs(model_train[1])
    threshold_ds_pairs = make_pairs(threshold_train[1])
    
    Xm,ym,_,_ = featurize_pairs(model_ds_pairs,model_train_features)
    Xth,yth,_,_ = featurize_pairs(threshold_ds_pairs,threshold_train_features)
    Xv,yv,_,_ = featurize_pairs(valid_pairs,valid_features)
    return Xm,ym,Xth,yth,Xv,yv
