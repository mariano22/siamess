from fastbook import *
from tqdm import tqdm

class SingleDS(torch.utils.data.Dataset):
    """ Dataset of single samples """
    def __init__(self, fns, get_x, get_y, tsfm):
        self.fns,self.get_x,self.get_y,self.tsfm = fns,get_x,get_y,tsfm
        self.cls = { y:i for i,y in enumerate(set(map(get_y,fns))) }

    def __getitem__(self, i):
        return ( self.tsfm(self.get_x(self.fns[i])), self.cls[self.get_y(self.fns[i])] )

    def __len__(self): return len(self.fns)

def get_embeddings(my_dl, m):
    with torch.no_grad():
        features_list = []
        y_list = []
        for x,y in tqdm(my_dl):
            features = m.encoder(x)
            features = nn.Flatten()(features)
            features_list.append(features)
            y_list.append(y)
        return torch.cat(features_list), torch.cat(y_list)

class Embeddings:
    def __init__(self, ds, name, bs=256):
        self.ds = ds
        self.dl = DataLoader(ds, batch_size=bs, shuffle=False, device=torch.device('cuda'))
        self.fn = f'embeddings-{name}.pt'

    def save(self):
        torch.save(self, self.fn)
        print(f'{self.fn} saved')
