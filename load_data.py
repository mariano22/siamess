from torchvision import transforms
from fastbook import *
import PIL
import cv2
from tqdm import tqdm
from datetime import datetime

class SiameseDataset(torch.utils.data.Dataset):
    """ Generic dataset for siamese image match problem """
    def __init__(self, fns, get_x, get_y, tsfm, valid_flag):
        self.fns = fns
        self.valid_flag = valid_flag
        self.get_x = get_x
        self.get_y = get_y
        self.tsfm = tsfm
        self.cls = [ get_y(f) for f in fns ]
        self.all_classes = set(self.cls)
        self.len = 2*len(fns)

        # Dict[ ClassLabel, List[Indexes] ] to easily pick class samples
        self.label_to_indexes = defaultdict(list)
        for i,l in enumerate(self.cls): self.label_to_indexes[l].append(i)

        # Precompute validation pairs for deterministic validation set
        if valid_flag: self.draw_memo = [ self._draw(i) for i in range(self.len) ]

    def __getitem__(self, i):
        idx1     = i//2
        is_match    = i%2
        idx2 = self.draw_memo[i] if self.valid_flag else self._draw(i)
        assert ((self.cls[idx1]==self.cls[idx2])==is_match)
        t1 = self.tsfm( self.get_x( self.fns[idx1] ) )
        t2 = self.tsfm( self.get_x( self.fns[idx2] ) )
        return ( t1, t2, torch.Tensor([is_match]).squeeze() )

    def __len__(self): return self.len

    def _draw(self, i):
        idx     = i//2
        is_match    = i%2
        if is_match: new_cls = self.cls[idx]
        else: new_cls = random.choice([l for l in self.all_classes if l != self.cls[idx]])
        return random.choice(self.label_to_indexes[new_cls])

class SiameseDatasetAllPairs(torch.utils.data.Dataset):
    """ Generic dataset for siamese image match problem (NOT USED, very slow) """
    def __init__(self, fns, get_x, get_y, tsfm):
        self.fns, self.get_x, self.tsfm = fns, get_x, tsfm
        self.cls = [ get_y(f) for f in fns ]

    def __getitem__(self, i):
        idx1     = i//len(self.fns)
        idx2     = i%len(self.fns)
        is_match = self.cls[idx1]==self.cls[idx2]
        t1 = self.tsfm( self.get_x( self.fns[idx1] ) )
        t2 = self.tsfm( self.get_x( self.fns[idx2] ) )
        return ( t1, t2, torch.Tensor([is_match]).squeeze() )

    def __len__(self): return len(self.fns)*len(self.fns)

def get_stats(fns, get_tensor, n_sample=1000):
    if n_sample!='all':
        fns = random.sample(fns, min(len(fns),n_sample))
    t = torch.stack([get_tensor(fn) for fn in fns])
    return t.mean(), t.std()

def calc_dss(path, get_items, get_x, get_y, is_valid, stats=None):
    """ Flexible workflow for constructing valid/train SiameseDataset """
    fns = get_items(path)
    print(f'{len(fns)} files loaded.')

    train_fns, valid_fns = [], []
    for fn in fns:
        if is_valid(fn):
            valid_fns.append(fn)
        else:
            train_fns.append(fn)
    print(f'{len(train_fns)} training samples / {len(valid_fns)} validation samples')

    if stats is None:
        stats = get_stats(train_fns, get_x)
        print(f'mean: {stats[0].item()} std: {stats[1].item()} (on train)')

    tsfm = transforms.Normalize(*stats)
    train_ds = SiameseDataset(train_fns, get_x, get_y, tsfm, valid_flag=False)
    valid_ds = SiameseDataset(valid_fns, get_x, get_y, tsfm, valid_flag=True)
    return train_ds, valid_ds

def show_siamese(t1, t2, l, more_info='', ctx=None, **kwargs):
    img = torch.cat([t1,t2], dim=2)
    match_msg = 'Undefined' if l is None else ['Not match','Match'][int(l)]
    title = ' '.join([match_msg, more_info])
    show_image(img, title=title, ctx=ctx, **kwargs)

# MNIST
RESIZE_SIZE=105
def get_x_mnist(fn):
    img=PIL.Image.open(fn)
    img=img.resize((RESIZE_SIZE,RESIZE_SIZE))
    _,img = cv2.threshold(cv2.GaussianBlur(np.array(img),(5,5),0),0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    img=PIL.Image.fromarray(img)
    img=PILImageBW(img)
    t = ToTensor()(img)
    t = (t>200).float()*255
    return t
def get_y_mnist(fn): return fn.parent.name
def is_validation_mnist(fn): return fn.parent.name in ['1','9']

# omniglot
def get_x_omniglot(fn):
    img = PILImageBW.create(fn)
    if img.shape!=(RESIZE_SIZE,RESIZE_SIZE):
        img=img.resize((RESIZE_SIZE,RESIZE_SIZE))
    t = ToTensor()( img )
    t = (255-t)/255.0
    return t
def get_y_omniglot(fn): return fn.parent.parent.name+'-'+fn.parent.name
def is_validation_omniglot(fn): return fn.parent.parent.parent.name=='images_evaluation'

# Sigcomp2009
def get_originals_train(path):
    fns = get_image_files(path)
    fns = [f for f in fns if is_original_sigcomp2009_train(f)]
    return fns

def get_originals_test(path):
    fns = get_image_files(path)
    fns = [f for f in fns if is_original_sigcomp2009_test(f)]
    return fns

def get_x_sigcomp2009(fn):
    img=PIL.Image.open(fn)
    img=img.resize((RESIZE_SIZE,RESIZE_SIZE))
    _,img = cv2.threshold(cv2.GaussianBlur(np.array(img),(1,1),0),0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    img=PIL.Image.fromarray(img)
    img=PILImageBW(img)
    t = ToTensor()(img).float()/255
    return t

def get_y_sigcomp2009_train(fn):
    return re.match(r'NISDCC-([0-9]{3})_([0-9]{3})_([0-9]{3})_6g.PNG',fn.name).group(2)
def get_y_sigcomp2009_test(fn):
    return re.match(r'NFI-([0-9]{3})([0-9]{2})([0-9]{3}).',fn.name).group(3)

def is_original_sigcomp2009_train(fn):
    m = re.match(r'NISDCC-([0-9]{3})_([0-9]{3})_([0-9]{3})_6g.PNG',fn.name)
    return m.group(1)==m.group(2)
def is_original_sigcomp2009_test(fn):
    m = re.match(r'NFI-([0-9]{3})([0-9]{2})([0-9]{3}).',fn.name)
    return m.group(1)==m.group(3)

def is_valid_sigcomp2009_train(fn):
    return any(va in fn.name for va in ['NISDCC-001', 'NISDCC-002'])

def sigcomp2009_train_or_test(fn,f_train,f_test):
    return f_train if fn.parent == path_sigcomp_train else f_test


# Data Loaders

path_omniglot = Path('../omniglot')
path_sigcomp = Path('../sigcomp')
path_sigcomp_train = Path('../sigcomp/preprocessed')
path_sigcomp_test = Path('../sigcomp/preprocessed-test')
path_mnist = untar_data(URLs.MNIST)

def save_wt(x,xn):
    ts = datetime.now().strftime("%d-%m-%Y-%H-%M-%S")
    fn = f'{xn}-{ts}.pt'
    print(f'Saving {xn} as {fn}')
    torch.save(x, fn)

def calc_dls_omniglot(bs=128, ds_fn=None):
    if ds_fn is None:
        train_ds, valid_ds = calc_dss(path_omniglot, get_image_files, get_x_omniglot, get_y_omniglot, is_validation_omniglot)
        save_wt((train_ds, valid_ds),'dss_omniglot')
    else:
        train_ds, valid_ds = torch.load(ds_fn)
    return DataLoaders.from_dsets(train_ds, valid_ds, bs=bs).cuda(), train_ds, valid_ds

def calc_dls_mnist(bs=128, ds_fn=None):
    if ds_fn is None:
        train_ds, valid_ds = calc_dss(path_mnist, get_image_files, get_x_mnist, get_y_mnist, is_validation_mnist)
    else:
        train_ds, valid_ds = torch.load(ds_fn)
    return DataLoaders.from_dsets(train_ds, valid_ds, bs=bs).cuda(), train_ds, valid_ds

def calc_dls_sigcomp_train(bs=32, ds_fn=None):
    if ds_fn is None:
        train_ds, valid_ds = calc_dss(path_sigcomp_train, get_originals_train, get_x_sigcomp2009, get_y_sigcomp2009_train, is_valid_sigcomp2009_train)
    else:
        train_ds, valid_ds = torch.load(ds_fn)
    return DataLoaders.from_dsets(train_ds, valid_ds, bs=bs).cuda(), train_ds, valid_ds

def get_image_files_truncated(length):
    def get_length_image_files(p):
        fns = get_image_files(p)
        random.shuffle(fns)
        return fns[:length]
    return get_length_image_files

def calc_ds_mnist(length, stats=None):
    fns = get_image_files_truncated(length)(path_mnist)
    if stats is None:
        stats = get_stats(fns, get_x_mnist)
    tsfm = transforms.Normalize(*stats)
    return SiameseDataset(fns, get_x_mnist, get_y_mnist, tsfm, valid_flag=True)

def calc_ds_sigcomp(dataset='train', stats=None):
    if dataset=='all':
        path = path_sigcomp
        get_items = lambda path: filter(lambda fn:
            is_original_sigcomp2009_train(fn) or is_original_sigcomp2009_test(fn),
            get_image_files(path))
        get_x = get_x_sigcomp2009
        get_y = lambda fn: sigcomp2009_train_or_test(fn, is_original_sigcomp2009_train, get_y_sigcomp2009_test)
    elif dataset=='train':
        path = path_sigcomp_train
        get_items = lambda path: list(filter(is_original_sigcomp2009_train, get_image_files(path)))
        get_x = get_x_sigcomp2009
        get_y = get_y_sigcomp2009_train
    elif dataset=='test':
        path = path_sigcomp_test
        get_items = lambda path: list(filter(is_original_sigcomp2009_test, get_image_files(path)))
        get_x = get_x_sigcomp2009
        get_y = get_y_sigcomp2009_test
    else: assert False
    fns = get_items(path)
    if stats is None:
        stats = get_stats(fns, get_x)
    tsfm = transforms.Normalize(*stats)
    return SiameseDataset(fns, get_x, get_y, tsfm, valid_flag=True)
