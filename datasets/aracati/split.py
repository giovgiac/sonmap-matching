# split.py

import glob
import pandas as pd
import numpy as np
import random as rand


PERCENT_TRAINING = 0.7
PERCENT_VALIDATION = 0.2
PERCENT_TESTING = 0.1

data = pd.read_csv("./data.csv")
son = sorted(glob.glob("./images/input/*.png"))
sat = sorted(glob.glob("./images/gt/*.png"))

indices = list(np.unique(data.iloc[:, 0]))
indices_size = len(indices)
assert(len(son) == len(sat) == len(indices))

# Acquire training data.
train_csv = pd.DataFrame()
for i in range(round(PERCENT_TRAINING * indices_size)):
    index = rand.choice(indices)
    train_csv = train_csv.append(data.iloc[:, :][data.iloc[:, 0] == index], ignore_index=True)
    indices.remove(index)

train_csv = train_csv.sort_values(by=['#Sonar ID'])
train_csv.to_csv(path_or_buf="./train/data.csv", index=False, header=True, float_format="%.6f")

# Acquire validation data.
valid_csv = pd.DataFrame()
for i in range(round(PERCENT_VALIDATION * indices_size)):
    index = rand.choice(indices)
    valid_csv = valid_csv.append(data.iloc[:, :][data.iloc[:, 0] == index], ignore_index=True)
    indices.remove(index)

valid_csv = valid_csv.sort_values(by=['#Sonar ID'])
valid_csv.to_csv(path_or_buf="./validation/data.csv", index=False, header=True, float_format="%.6f")

# Acquire testing data.
test_csv = pd.DataFrame()
for i in range(round(PERCENT_TESTING * indices_size)):
    index = rand.choice(indices)
    test_csv = test_csv.append(data.iloc[:, :][data.iloc[:, 0] == index], ignore_index=True)
    indices.remove(index)

test_csv = test_csv.sort_values(by=['#Sonar ID'])
test_csv.to_csv(path_or_buf="./test/data.csv", index=False, header=True, float_format="%.6f")
