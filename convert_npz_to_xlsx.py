import numpy as np
import pandas as pd
import os

npz_file = np.load('dataset/input.npz')
for k in npz_file.keys():
    print(k)
    # Make 3D to 2D
    npy_file = npz_file[k]
    print(npy_file.shape)
    npy_file = np.reshape(npy_file, (npy_file.shape[-1], -1))
    npy_file = np.transpose(npy_file)
    print(npy_file.shape)

    # Make directory
    os.makedirs(f'dataset/generated', exist_ok=True)

    # Save to xlsx
    df = pd.DataFrame(npy_file)
    df.to_excel(f'dataset/generated/{k}.xlsx')