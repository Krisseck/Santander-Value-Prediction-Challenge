# Given the statistics for past 3 hours,
# make traffic predictions for the next 4 hours

import numpy as np
import random
import math
from keras.models import Sequential, load_model
from keras.layers import Conv1D, MaxPooling1D, GlobalAveragePooling1D, Embedding, LSTM, Dropout, Dense, Flatten
from keras.constraints import max_norm
from keras.optimizers import Adam
import keras.backend as K

epochs = 5
batch_size = 4

source_csv = 'train.csv'
source_csv_delimiter = ','

# fix random seed for reproducibility
np.random.seed(7)

source_data = np.genfromtxt(source_csv, delimiter=source_csv_delimiter, usecols=range(1,4993), skip_header=1, dtype='float_')

# Get training and test data

trainX = np.zeros(shape=(source_data.shape[0],4991))
trainY = np.zeros(shape=(source_data.shape[0],1))

# Real naive, just divide everything by 10^9

source_data /= 1000000000

for i, row in enumerate(source_data):
  trainX[i] = row[1:]
  # the scale for target is 10^7
  trainY[i] = row[0] * 100

# make it divisable by batch size
remainder = len(trainX) % batch_size
if remainder > 0:
  trainX = trainX[:-remainder]
  trainY = trainY[:-remainder]

print(trainX.shape)
print(trainY.shape)

print(trainY[10])

# create and fit model
model = Sequential()

# Got 2.0 on Kaggle, loss: 0.4614 - val_loss: 0.4303
'''
trainX = np.expand_dims(trainX, axis=2)

model.add(Conv1D(input_shape=(trainX.shape[1], 1), filters=200, kernel_size=8, activation='relu'))
model.add(MaxPooling1D(4))
model.add(Dropout(0.5))
model.add(Flatten())
model.add(Dense(100, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))
'''

model.add(Dense(1000, input_shape=(trainX.shape[1], ), activation='linear'))
model.add(Dropout(0.5))
model.add(Dense(5000, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1, activation='linear'))

def root_mean_squared_logarithmic_error(y_true, y_pred):
    y_pred_log = K.log(K.clip(y_pred, K.epsilon(), None) + 1.)
    y_true_log = K.log(K.clip(y_true, K.epsilon(), None) + 1.)
    return K.sqrt(K.mean(K.square(y_pred_log - y_true_log), axis = -1))

model.compile(loss='mae', optimizer='adam')

while True:
  model.fit(trainX, trainY, epochs=epochs, batch_size=batch_size, validation_split=0.1)
  model.save('model.h5')
