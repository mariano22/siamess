from torchvision import transforms
from fastbook import *
import PIL
import cv2
from tqdm import tqdm
from datetime import datetime


class SiameseImage(fastuple):
    def show(self, ctx=None, **kwargs):
        if len(self) > 2:
            img1,img2,similarity = self
            match_msg = ['Not match','Match'][int(similarity)]
        else:
            img1,img2 = self
            similmatch_msgarity = 'Undetermined'
        if not isinstance(img1, Tensor):
            if img2.size != img1.size: img2 = img2.resize(img1.size)
            t1,t2 = tensor(img1),tensor(img2)
            if t1.ndim!=3: t1 = t1.unsqueeze(2)
            if t2.ndim!=3: t2 = t2.unsqueeze(2)
            t1,t2 = t1.permute(2,0,1),t2.permute(2,0,1)
        else: t1,t2 = img1,img2
        line = t1.new_zeros(t1.shape[0], t1.shape[1], 10)
        return show_image(torch.cat([t1,line,t2], dim=2), title=match_msg, ctx=ctx, **kwargs)

@typedispatch
def show_batch(x:SiameseImage, y, samples, ctxs=None, max_n=6, nrows=None, ncols=2, figsize=None, **kwargs):
    if figsize is None: figsize = (ncols*6, max_n//ncols * 3)
    if ctxs is None: ctxs = get_grid(min(x[0].shape[0], max_n), nrows=None, ncols=ncols, figsize=figsize)
    for i,ctx in enumerate(ctxs): SiameseImage(x[0][i], x[1][i], x[2][i].item()).show(ctx=ctx)

class SiameseDataset(torch.utils.data.Dataset):
    """ Generic dataset for siamese image match problem """
    def __init__(self, fns, get_x, get_y, valid_flag):
        self.fns = fns
        self.valid_flag = valid_flag
        self.get_x = get_x
        self.get_y = get_y
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
        x1 = self.get_x( self.fns[idx1] )
        x2 = self.get_x( self.fns[idx2] )
        return SiameseImage(x1, x2, torch.Tensor([is_match]).squeeze())

    def __len__(self): return self.len

    def _draw(self, i):
        idx     = i//2
        is_match    = i%2
        if is_match: new_cls = self.cls[idx]
        else: new_cls = random.choice([l for l in self.all_classes if l != self.cls[idx]])
        return random.choice(self.label_to_indexes[new_cls])

class SiameseDatasetAllPairs(torch.utils.data.Dataset):
    """ Generic dataset for siamese image match problem (NOT USED, very slow) """
    def __init__(self, fns, get_x, get_y):
        self.fns, self.get_x = fns, get_x
        self.cls = [ get_y(f) for f in fns ]

    def __getitem__(self, i):
        idx1     = i//len(self.fns)
        idx2     = i%len(self.fns)
        is_match = self.cls[idx1]==self.cls[idx2]
        x1 = self.get_x( self.fns[idx1] )
        x2 = self.get_x( self.fns[idx2] )
        return SiameseImage(x1, x2, torch.Tensor([is_match]).squeeze())

    def __len__(self): return len(self.fns)*len(self.fns)

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

    train_ds = SiameseDataset(train_fns, get_x, get_y, valid_flag=False)
    valid_ds = SiameseDataset(valid_fns, get_x, get_y, valid_flag=True)
    return train_ds, valid_ds
"""
if stats is None:
        stats = get_stats(train_fns, get_x)
        print(f'mean: {stats[0].item()} std: {stats[1].item()} (on train)')

    tsfm = transforms.Normalize(*stats)
"""

# Data Loaders

def save_wt(x,xn):
    ts = datetime.now().strftime("%d-%m-%Y-%H-%M-%S")
    fn = f'{xn}-{ts}.pt'
    print(f'Saving {xn} as {fn}')
    torch.save(x, fn)


def get_image_files_truncated(length):
    def get_length_image_files(p):
        fns = get_image_files(p)
        random.shuffle(fns)
        return fns[:length]
    return get_length_image_files
