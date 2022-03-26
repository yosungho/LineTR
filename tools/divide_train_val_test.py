import os
import numpy as np
from pathlib import Path
from tqdm import tqdm

def split_train_val(X, val_size=0.3, shuffle=True, random_state=1004):
    test_num = int(len(X) * val_size)
    train_num = len(X) - test_num

    if shuffle:
        np.random.seed(random_state) # for reproducibility
        np.random.shuffle(X)           # random shuffling second
        X_train = X[:train_num]  # slicing from 1 to train_num row, X column
        X_val = X[train_num:]   # slicing from 1 to train_num row, y column
    else:
        X_train = X[:train_num]
        X_val = X[train_num:]
    
    return X_train, X_val

from_path = Path('assets/dataset/dataset_h5')
to_path = Path('assets/dataset/dataset_h5')
train_path = to_path / 'train'
val_path = to_path / 'val'
# test_path = to_path / 'test'

os.system('rm -r %s' %train_path)
os.system('rm -r %s' %val_path)
# os.system('rm -r %s' %test_path)
os.mkdir(train_path)
os.mkdir(val_path)
# os.mkdir(test_path)

listing = sorted(Path(from_path).glob('*.h5'))
path_train, path_val = split_train_val(listing[:], val_size=0.2)
# path_test, path_val = split_train_val(path_val[:], val_size=0.5) 

for file in tqdm(path_train):
    file.stem
    os.system('cp -r %s %s'%(file, train_path))

for file in tqdm(path_val):
    file.stem
    os.system('cp -r %s %s'%(file, val_path))

# for file in tqdm(path_test):
#     file.stem
#     os.system('cp -r %s %s'%(file, test_path))
    
print('done')