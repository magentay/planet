import pandas as pd
import h5py
import numpy as np
from tqdm import tqdm
import cv2

trainpath = '/data/planet/train-jpg/'
testpath = '/data/planet/test-jpg/'

datadir = '/data/planet/features/'

x_train = pd.read_csv('/data/planet/train_v2.csv')

flatten = lambda l: [item for sublist in l for item in sublist]
labels = sorted(list(set(flatten([l.split(' ') for l in x_train['tags'].values]))))

label_map = {l: i for i, l in enumerate(labels)}
inv_label_map = {i: l for l, i in label_map.items()}
print(label_map)
print(inv_label_map)

y_train = []
for f, tags in tqdm(x_train.values, miniters=1000):
    targets = np.zeros(17)
    for t in tags.split(' '):
        targets[label_map[t]] = 1
    y_train.append(targets)
y_train = np.array(y_train, np.uint8)


input_dim = 64
#===========train set ===============
hdf5_path = '/data/planet/h5features/train_' + str(input_dim)+'x'+str(input_dim)+'.h5'
hdf5_file = h5py.File(hdf5_path, mode = 'w')
train_shape = (len(x_train),input_dim, input_dim, 3)
hdf5_file.create_dataset('x_train', train_shape, np.float32)
hdf5_file.create_dataset('x_train_flipx',train_shape, np.float32)
hdf5_file.create_dataset('x_train_flipy',train_shape, np.float32)
hdf5_file.create_dataset('x_train_flipz',train_shape, np.float32)
hdf5_file.create_dataset('y_train', data=y_train)
trainlen = len(x_train)

for i in tqdm(range(trainlen), miniters=10):


    f = x_train.image_name.values[i]

    img = cv2.imread('/data/planet/train-jpg/{}.jpg'.format(f))

    img64 = cv2.resize(img, (input_dim, input_dim))
    imgx = cv2.flip(img64, 0)
    imgy = cv2.flip(img64, 1)
    imgz = cv2.flip(img64,-1)


    hdf5_file["x_train"][i,...] = img64[None]/255.
    hdf5_file["x_train_flipx"][i, ...] = imgx[None]/255.
    hdf5_file["x_train_flipy"][i , ...] = imgy[None]/255.
    hdf5_file["x_train_flipz"][i , ...] = imgz[None]/255.



hdf5_file.close()

#=============== test set =======================
x_test = pd.read_csv('/data/planet/sample_submission_v2.csv')
hdf5_path = '/data/planet/h5features/test_' + str(input_dim)+'x'+str(input_dim)+'.h5'
hdf5_file = h5py.File(hdf5_path, mode = 'w')
test_shape = (len(x_test),input_dim, input_dim, 3)

hdf5_file.create_dataset('x_test', test_shape, np.float32)
for i in tqdm(range(len(x_test)), miniters=10):
    f = x_test.image_name.values[i]
    img = cv2.imread('/data/planet/test-jpg/{}.jpg'.format(f))
    img64 = cv2.resize(img, (input_dim, input_dim))
    hdf5_file["x_test"][i, ...] = img64[None]/255.
hdf5_file.close()