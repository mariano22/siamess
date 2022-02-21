from libraries_import_all import *

def get_elements(path, get_files, get_cls, is_valid):
    if not isinstance(path,list): path = [path]
    fns = [ f for p in path for f in get_files(p) ]
    train_fns, valid_fns = L(f for f in fns if not is_valid(f)), L(f for f in fns if is_valid(f))
    train_idx2cls, valid_idx2cls = L(map(get_cls,train_fns)),L(map(get_cls,valid_fns))
    return (train_fns, train_idx2cls), (valid_fns, valid_idx2cls)

class PairDS:
    def __init__(self, idx2cls, cls_sampling_limit=None):
        cls = L(set(idx2cls))
        
        cls2idx = defaultdict(list)
        for (i,y) in enumerate(idx2cls): cls2idx[y].append(i)

        if cls_sampling_limit is None: cls_sampling_limit = min(list(Counter(idx2cls).values()))
        # Authentic pairs
        authentics_pairs = [ (xi,xj,True) for y in cls 
                                  for i,xi in enumerate(cls2idx[y]) 
                                  for j,xj in enumerate(cls2idx[y][i+1:]) if j<cls_sampling_limit ]
        # False pairs
        false_n_sampling = [ (xi,len(cls2idx[y])-i-1) for y in cls for i,xi in enumerate(cls2idx[y]) ]
        def draw_false(x):
            c = random.choice([l for l in cls if l!=idx2cls[x]])
            return random.choice(cls2idx[c])
        random_false = [ (xi,draw_false(xi),False) for xi, n_false in false_n_sampling for _ in range(n_false) ]
        
        pairs = random_false+authentics_pairs
        random.shuffle(pairs)
        pairs = L(pairs)
        store_attr('idx2cls,cls2idx,cls,pairs', self)