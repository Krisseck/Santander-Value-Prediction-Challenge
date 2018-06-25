# Given the statistics for past 3 hours,
# make traffic predictions for the next 4 hours

import numpy as np
import random
import math
from keras.models import Sequential, load_model
from keras.layers import Conv1D, MaxPooling1D, GlobalAveragePooling1D, Embedding, LSTM, Dropout, Dense, Flatten
from keras.constraints import max_norm
import time

epochs = 5
batch_size = 20

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
  trainY[i] = row[0]

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

trainX = np.expand_dims(trainX, axis=2)

model.add(Dense(1000, input_shape=(trainX.shape[1], 1), activation='relu'))
model.add(Flatten())
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='mae', optimizer='adam')

def root_mean_squared_error(y_true, y_pred):
  return K.sqrt(K.mean(K.square(y_pred - y_true), axis=-1)) 

while True:
  model.fit(trainX, trainY, epochs=epochs, batch_size=batch_size, validation_split=0.2)
  model.save('model.h5')
