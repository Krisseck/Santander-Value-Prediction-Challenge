import numpy as np
import random
import math
import time
import pandas as pd
from keras.models import load_model

epochs = 5
batch_size = 20
chunk_size = 1000

source_csv = 'test.csv'
source_csv_delimiter = ','

# fix random seed for reproducibility
np.random.seed(7)

model = load_model('model.h5')

print("ID,target")

for chunk in pd.read_csv(source_csv, chunksize=chunk_size, header=0, skiprows=0):
  input_rows = []
  for index, row in chunk.iterrows():
    input_row = row[1:]
    input_row /= 1000000000
    input_rows.append(input_row)
  scores = model.predict(np.array(input_rows))
  i = 0
  for index, row in chunk.iterrows():
    print(row[0]+","+repr(scores[i][0] * 1000000000))
    i += 1