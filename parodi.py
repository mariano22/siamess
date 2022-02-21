from libraries_import_all import *

from features_grid import *

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

def load_features(in_name,ndiv):
    OUT_PATH = Path('./data/parodi_out/features')/Path(str(ndiv))
    with open(OUT_PATH/Path(in_name+'.npy'), 'rb') as f:
        x = np.load(f)
    with open(OUT_PATH/Path(in_name+'.txt'), 'r') as f:
        fns = L(f.readlines()).map(lambda x : x.rstrip())
    return fns,x
