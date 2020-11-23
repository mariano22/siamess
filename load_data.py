from fastbook import *

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

def get_stats(fns, get_tensor, n_sample=1000):
    fns = random.sample(fns, min(len(fns),n_sample))
    t = torch.stack([get_tensor(fn) for fn in fns])
    return t.mean(), t.std()

def calc_dss(path, get_items, get_x, get_y, is_valid):
    """ Flexible workflow for constructing valid/train SiameseDataset """
    fns = get_items(path)
    print(f'{len(fns)} files loaded.')

    train_fns, valid_fns = [], []
    for fn in fns:
        if is_valid(fn):
            train_fns.append(fn)
        else:
            valid_fns.append(fn)
    print(f'{len(train_fns)} training samples / {len(valid_fns)} validation samples')

    m,s = get_stats(train_fns, get_x)
    print(f'mean: {m.item()} std: {s.item()} (on train)')

    tsfm = transforms.Normalize(m,s)
    train_ds = SiameseDataset(train_fns, get_x, get_y, tsfm, valid_flag=False)
    valid_ds = SiameseDataset(valid_fns, get_x, get_y, tsfm, valid_flag=True)
    return train_ds, valid_ds

def show_siamese(t1, t2, l, more_info='', ctx=None, **kwargs):
    img = torch.cat([t1,t2], dim=2)
    match_msg = 'Undefined' if l is None else ['Not match','Match'][int(l)]
    title = ' '.join([match_msg, more_info])
    show_image(img, title=title, ctx=ctx, **kwargs)

# MNIST
def get_x_mnist(fn):
    img = PILImageBW.create(fn)
    img = PILImageBW(img.resize((105,105)))
    t = ToTensor()(img)
    t = (t>200).float()*255
    return t
def get_y_mnist(fn): return fn.parent.name
def is_validation_mnist(fn): return f.parent.name in ['1','9']

# omniglot
def get_x_omniglot(fn): return ToTensor()( PILImageBW.create(fn) )/255.0
def get_y_omniglot(fn): return fn.parent.parent.name+'-'+fn.parent.name
def is_validation_omniglot(fn): return fn.parent.parent.parent.name=='images_background'
