from libraries_import_all import *

from tools import *

def get_elements(path, get_files, get_cls, is_valid):
    """ Returns a pair train, test. Each one is a tuple (fns, cls) where fns are the image files and cls are the respective classes. """
    if not isinstance(path,list): path = [path]
    fns = [ f for p in path for f in get_files(p) ]
    train_fns, valid_fns = L(f for f in fns if not is_valid(f)), L(f for f in fns if is_valid(f))
    train_idx2cls, valid_idx2cls = L(map(get_cls,train_fns)),L(map(get_cls,valid_fns))
    return (train_fns, train_idx2cls), (valid_fns, valid_idx2cls)

def make_pairs(idx2cls, cls_sampling_limit=None):
    """  Given idx2cls (the classes list) construct list of pairs (xi,xj,True) where xi and xj are indices and True if the pair is authentic. """
    cls = L(set(idx2cls))
    
    cls2idx = defaultdict(L)
    for (i,y) in enumerate(idx2cls): cls2idx[y].append(i)

    if cls_sampling_limit is None: cls_sampling_limit = min(list(Counter(idx2cls).values()))
    # Authentic pairs
    authentics_pairs = [ (xi,xj,True) for y in cls 
                                for i,xi in enumerate(cls2idx[y]) 
                                for j,xj in enumerate(cls2idx[y][i+1:]) if j<cls_sampling_limit ]
    # False pairs
    false_n_sampling = [ (xi, min(cls_sampling_limit,len(cls2idx[y])-i-1)) for y in cls for i,xi in enumerate(cls2idx[y]) ]
    def draw_false(x):
        c = random.choice([l for l in cls if l!=idx2cls[x]])
        return random.choice(cls2idx[c])
    random_false = [ (xi,draw_false(xi),False) for xi, n_false in false_n_sampling for _ in range(n_false) ]
    
    pairs = random_false+authentics_pairs
    random.shuffle(pairs)
    return L(pairs)

def experiment_dataset(dev,test,cls_sampling_limit_test=None):
    """ Given a dev and test (tuples of (fns, cls)) show some metrics and fix the test (list of 3-uples (idx_1, idx_2, bool)). """
    dev_len, test_len = len(dev[0]),len(test[0])
    dev_cls_len, test_cls_len = len(set(dev[1])), len(set(test[1]))
    print('N dev files: {} | N test files {} | files test/ratio: {}'.format(dev_len, test_len, test_len/(dev_len+test_len)))
    print('N dev classes: {} | N test classes: {} | classes test/ratio: {}'.format(dev_cls_len, test_cls_len, test_cls_len/(dev_cls_len+test_cls_len)))
    print('cls_sampling_limit_test = {}'.format(cls_sampling_limit_test))
    print('Min class size dev: {} | Min class size test: {}'.format(min(Counter(dev[1]).values()), min(Counter(test[1]).values())))
    print('Max class size dev: {} | Max class size test: {}'.format(max(Counter(dev[1]).values()), max(Counter(test[1]).values())))
    test_ds = make_pairs(test[1],cls_sampling_limit_test)
    print('N pairs test: {} | true ratio: {}'.format(len(test_ds), sum(1 for _,_,y in test_ds if y)/len(test_ds)))
    return dev,test,test_ds

def test_solution(experiment,solution):
    x1,x2,y = zip(*experiment.test_ds.pairs)
    x = L(zip(experiment.test[0][x1], experiment.test[0][x2]))
    pred=solution.predict(x)
    sts = calc_stats(pred,y)
    print_stats(sts)
    return sts

def cross_validation_slices(L,K):
    delta = int( L / K )
    return [ slice(i*delta, (i+1)*delta) for i in range(K-1) ] + [ slice((K-1)*delta , L) ]

def apply_split(split_idxs,l):
    l = split_idxs.map(lambda i: (l[i.map(lambda x : not x)], l[i]))
    return L(zip(*l))

def assert_mutual_disjoint(clss):
    for i in range(len(clss)):
        for j in range(i+1,len(clss)):
            assert 0 == len( set(clss[i]).intersection(set(clss[j])) )

def make_cv_splits(dev):
    fns,cls = dev
    cls_set = L(set(cls))
    cv_slices = cross_validation_slices(len(cls_set),6)

    cls_folds = L( cls_set[s] for s in cv_slices)
    assert_mutual_disjoint(cls_folds)

    split_idxs = cls_folds.map(lambda f : cls.map(lambda c : c in f) )
    train_fns, valid_fns = apply_split(split_idxs, fns)
    train_cls, valid_cls = apply_split(split_idxs, cls)

    train = L(zip(train_fns,train_cls))
    valid = L(zip(valid_fns,valid_cls))
    split = L(zip(train,valid))
    print( 'N splits train/validation:' + ' '.join('({}/{})'.format(len(t_fns),len(v_fns)) for (t_fns, _),(v_fns, _) in split) )
    print( 'N classes train/validation:' + ' '.join('({}/{})'.format(len(set(t_cls)),len(set(v_cls))) for (_, t_cls),(_, v_cls) in split) )
    for (t_fns, t_cls),(v_fns, v_cls) in split:
        assert len(t_fns)==len(t_cls)
        assert len(v_fns)==len(v_cls)
    return split
class Experiment:
    """ Experiment setting: 
        - dev is for training (fns, cls). 
        - test (fns, cls) and test_ds (i,j,True) is for evaluating. """
    def __init__(self,dev,test,cls_sampling_limit_validation=None, cls_sampling_limit_test=None):
        dev,test,test_ds = experiment_dataset(dev,test,cls_sampling_limit_test)
        split = make_cv_splits(dev)
        split_valid_ds = L( make_pairs(v_cls,cls_sampling_limit_validation) for _,(_,v_cls) in split )
        print('Len split pairs: {}'.format(split_valid_ds.map(len)))
        print('true ratio split pairs: {}'.format(split_valid_ds.map(lambda ds : sum(1 for _,_,y in ds if y)/len(ds))))
        store_attr('dev,test,test_ds,split,split_valid_ds', self)

class Solution:
    def __init__(self, fit, predict): store_attr('fit,predict', self)
