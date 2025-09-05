import pandas as pd
import numpy as np
import os

data_path = ""
Antibod_train   = pd.read_csv(os.path.join("./", 'train.s004.csv'), low_memory=False, sep=',')
Antibod_test    = pd.read_csv(os.path.join("./", 'test.s005.csv'), low_memory=False, sep=',')

threshold = 5
Antibod_test['Immun_label'] = np.where(Antibod_test['Immun_value'] < threshold, 0, 1)
Antibod_test.to_csv('test_5%.csv', index=False)

Antibod_train['Immun_label'] = np.where(Antibod_train['Immun_value'] < threshold, 0, 1)
Antibod_train.to_csv('train_5%.csv', index=False)

