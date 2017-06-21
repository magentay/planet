import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import gc
import pickle
import keras as k
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from sklearn.model_selection import train_test_split
from keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import fbeta_score
import keras.backend as K
from planet_tools import *
import os.path


from tqdm import tqdm
datadir = '/data/planet/features/'
modeldir ='/data/planet/models/'
input_dim = 64
conv1 = 32
conv2 = 64
conv3 = 128
epoch = 10

best_weights_filepath = modeldir+ str(conv1)+str(conv2)+str(conv3)+'_best_'+str(input_dim)+'x'+str(input_dim)+'_weights.hdf5'
train_hdf5_path = '/data/planet/features/train_' + str(input_dim) + 'x' + str(input_dim) + '.h5'
valid_hdf5_path = '/data/planet/features/valid_' + str(input_dim) + 'x' + str(input_dim) + '.h5'
test_hdf5_path = '/data/planet/features/test_' + str(input_dim) + 'x' + str(input_dim) + '.h5'

#==============================================================================
model = Sequential()
model.add(Conv2D(conv1, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=(input_dim,input_dim, 3)))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(conv2, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(conv3, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(17, activation='sigmoid'))


earlyStopping = EarlyStopping(monitor='val_acc', patience=2, verbose=0, mode='auto')
saveBestModel = ModelCheckpoint(best_weights_filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max', period=1)

model.compile(loss='binary_crossentropy',
              # We NEED binary here, since categorical_crossentropy l1 norms the output before calculating loss.
              optimizer='adam',
              metrics=['accuracy',fbeta],
              )
print(model.summary())
############################################################

if os.path.isfile(best_weights_filepath):
    model.load_weights(best_weights_filepath)


x_train = HDF5Matrix(train_hdf5_path, 'x_train')
y_train = HDF5Matrix(train_hdf5_path, 'y_train')
x_valid = HDF5Matrix(valid_hdf5_path, 'x_valid')
y_valid = HDF5Matrix(valid_hdf5_path, 'y_valid')

model.fit(x_train, y_train,
          batch_size=128,
          epochs=epoch,
          verbose=1,
          validation_data=(x_valid, y_valid),
          callbacks=[saveBestModel])

##############################################################################
p_valid = model.predict(x_valid, batch_size=128)

hdf5_file = h5py.File(valid_hdf5_path, mode = 'r')
y_valid = hdf5_file["y_valid"][...]
hdf5_file.close()

bestbeta, bestfbeta = pickBeta(y_valid[:,:], p_valid[:,:])
outfile = '/data/planet/results/keras_'+str(input_dim)+'x'+str(input_dim)+'_epoch'+str(epoch)+"_"+str(bestfbeta)+'.csv'
predict_test_h5py(model,test_hdf5_path,bestbeta,outfile )

##########################################
model.load_weights(best_weights_filepath)
p_valid = model.predict(x_valid, batch_size=128)

bestbeta, bestfbeta = pickBeta(y_valid[:,:], p_valid[:,:])

outfile = '/data/planet/results/keras_'+str(input_dim)+'x'+str(input_dim)+'_best_epoch'+str(epoch)+"_"+str(bestfbeta)+'.csv'
predict_test_h5py(model, test_hdf5_path,bestbeta,outfile )
