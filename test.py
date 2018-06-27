import numpy as np
import random
import math
import time
import pandas as pd
from keras.models import load_model
import keras.backend as K

epochs = 5
batch_size = 20
chunk_size = 4000

source_csv = 'test.csv'
source_csv_delimiter = ','

# fix random seed for reproducibility
np.random.seed(7)

def root_mean_squared_logarithmic_error(y_true, y_pred):
    y_pred_log = K.log(K.clip(y_pred, K.epsilon(), None) + 1.)
    y_true_log = K.log(K.clip(y_true, K.epsilon(), None) + 1.)
    return K.sqrt(K.mean(K.square(y_pred_log - y_true_log), axis = -1))

model = load_model('model.h5')

print("ID,target")

for chunk in pd.read_csv(source_csv, chunksize=chunk_size, header=0, skiprows=0):
  input_rows = []
  for index, row in chunk.iterrows():
    input_row = row[1:]
    input_row /= 1000000000
    input_rows.append(input_row)
  input_rows = np.array(input_rows)
  scores = model.predict(input_rows)
  i = 0
  for index, row in chunk.iterrows():
    print(row[0]+","+repr(scores[i][0] * 10000000))
    i += 1
