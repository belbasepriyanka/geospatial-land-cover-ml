import numpy as np

def spatial_block_ids(x, y, block_size=20.0):
    x = np.asarray(x)
    y = np.asarray(y)
    bx = np.floor((x - x.min()) / block_size).astype(int)
    by = np.floor((y - y.min()) / block_size).astype(int)
    return bx.astype(str) + "_" + by.astype(str)
