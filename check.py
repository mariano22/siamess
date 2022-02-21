from import_all import *

def idx_dict_to_numpy(a): return np.array([a[i] for i in range(len(a))])

""" Load all the train images. """
PREPROCESSED_PATH='./data/parodi_out/preproc'
get_fns = ds_sigcomp2009.get_authentic_signatures
get_x = ds_sigcomp2009.get_x
preprocess_dataset = L(get_fns(PREPROCESSED_PATH))

for f in preprocess_dataset:
    print(f)
    fgrid = FastGrid(f, ds_sigcomp2009.get_x, N_divisions=16)
    grid = Grid(f, N_divisions=16)
    assert(grid.N_divisions == fgrid.N_divisions )
    assert(grid.N == fgrid.N )
    assert(grid.M == fgrid.M )
    assert((grid.center == fgrid.center).all())
    assert( (idx_dict_to_numpy(grid.idx_to_center)==fgrid.idx_to_center).all() )
    assert( (idx_dict_to_numpy(grid.idx_to_pixel_count)==fgrid.idx_to_pixel_count).all() )
    assert( (idx_dict_to_numpy(grid.idx_to_dist)==fgrid.idx_to_dist).all() )
    assert( (idx_dict_to_numpy(grid.idx_to_vector_end)==fgrid.idx_to_vector_end).all() )
    assert( (idx_dict_to_numpy(grid.idx_to_angle)==fgrid.idx_to_angle).all() )
    assert( ( np.array(grid.farest) == np.array(fgrid.farest) ).all() )
    assert( grid.R == fgrid.R )
    print("ok")
