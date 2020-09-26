import  keras
from sklearn.model_selection import  train_test_split
import cv2

TEST_DIR='./data/test/'

SIGNATURE_CLASSES = ['Genuine','Forged']#,'C','D','E','F','G','H','I','J','K','L','M']

import os, random
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss
from sklearn.preprocessing import LabelEncoder

import matplotlib.pyplot as plt
from matplotlib import ticker
#import seaborn as sns
#%matplotlib inline 

from keras.models import Sequential
from keras.layers import Dropout, Flatten, Conv2D, MaxPooling2D, ZeroPadding2D, Dense, Activation
from keras.optimizers import SGD, Adagrad
from keras.callbacks import EarlyStopping
from keras.utils import np_utils
from keras.optimizers import RMSprop, Adam
from keras import backend as K
ROWS = 190  
COLS = 160 
CHANNELS = 3
TRAIN_DIR="./data/train/"

def root_mean_squared_error(y_true, y_pred):
    """
    RMSE loss function
    """
    return K.sqrt(K.mean(K.square(y_pred - y_true), axis=-1))


def get_images(fish):
    """Load files from train folder"""
    fish_dir = TRAIN_DIR+'{}'.format(fish)
    images = [fish+'/'+im for im in os.listdir(fish_dir)]
    return images

def read_image(src):
    """Read and resize individual images"""
    im = cv2.imread(src, cv2.IMREAD_COLOR)
    im = cv2.resize(im, (COLS, ROWS), interpolation=cv2.INTER_CUBIC)
    return im
"""def read_image(src):
    import os
    from scipy import misc
    import scipy.misc  as mc
    filepath=src
    im=misc.imread(filepath)
     
    return mc.imresize(im,(ROWS,COLS))"""
#grandPadre=read_gray_scale()


files = []
y_all = []

for fish in SIGNATURE_CLASSES:
    fish_files = get_images(fish)
    files.extend(fish_files)
    
    y_fish = np.tile(fish, len(fish_files))
    y_all.extend(y_fish)
    print("{0} photos of {1}".format(len(fish_files), fish))
    
y_all = np.array(y_all)
print(len(files))
print(len(y_all))




X_all = np.ndarray((len(files), ROWS, COLS, CHANNELS), dtype=np.uint8)

for i, im in enumerate(files): 
    X_all[i] = read_image(TRAIN_DIR+im)
    if i%1000 == 0: print('Processed {} of {}'.format(i, len(files)))

print(X_all.shape)
# One Hot Encoding Labels
y_all = LabelEncoder().fit_transform(y_all)
y_all = np_utils.to_categorical(y_all)




from sklearn.model_selection import train_test_split

X_train, X_valid, y_train, y_valid = train_test_split(X_all, y_all, 
                                                    test_size=2, random_state=23, 
                                                    stratify=y_all)



optimizer = RMSprop(lr=1e-4)
objective = 'categorical_crossentropy'
def center_normalize(x):
    return (x - K.mean(x)) / K.std(x)
print('1')
model = Sequential()

model.add(Activation(activation=center_normalize, input_shape=(ROWS, COLS, CHANNELS)))

model.add(Conv2D(64, (3, 3), padding='same'))
model.add(Activation('relu'))
model.add(Conv2D(64, (3, 3), padding='valid'))
model.add(Activation('relu'))
model.add(ZeroPadding2D(padding=(1, 1)))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(96, (3, 3), padding='same'))
model.add(Activation('relu'))
model.add(Conv2D(96, (3, 3), padding='valid'))
model.add(Activation('relu'))
model.add(ZeroPadding2D(padding=(1, 1)))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(128, (3, 3), padding='same'))
model.add(Activation('relu'))
model.add(Conv2D(128, (3, 3), padding='same'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(1024))
model.add(Activation('relu'))
model.add(Dropout(0.5))


model.add(Dense(len(SIGNATURE_CLASSES)))
model.add(Activation('sigmoid'))
 
adam = Adam(lr=0.0001)
model.compile(optimizer=adam, loss=root_mean_squared_error)


early_stopping = EarlyStopping(monitor='val_loss', patience=4, verbose=1, mode='auto')        
       
model.fit(X_train, y_train, batch_size=64, epochs=4,
              validation_split=0.1, verbose=1, shuffle=True, callbacks=[early_stopping])
preds = model.predict(X_valid, verbose=1)
print("Validation Log Loss: {}".format(log_loss(y_valid, preds)))




test_files = [im for im in os.listdir(TEST_DIR)]
test = np.ndarray((len(test_files), ROWS, COLS, CHANNELS), dtype=np.uint8)

for i, im in enumerate(test_files): 
    test[i] = read_image(TEST_DIR+im)
    
test_preds = model.predict(test, verbose=1)

fList = test_preds.tolist()
for f in fList:
    if f[0] > f[1]:
        f.append("Forged")
    else:
        f.append("Genuine")
fList = np.array(fList)

submission = pd.DataFrame(fList, columns=['Genuine','Forged','Results'])
submission.insert(0, 'image', test_files)
submission.head()

submission.to_csv('./signatureResults.csv',index=False)
