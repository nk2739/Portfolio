############################
#                          #
# authors:                 #
# Zixi Huang(zh2313)       # 
# Neil Kumar(nk2739)       # 
# Yichen Pan(yp2450)       #
#                          #
############################

from scipy.io import loadmat
from sklearn.linear_model import SGDRegressor
import numpy as np

# load data
data = loadmat("/Users/cee/Downloads/MSdata.mat")

x_test = data["testx"]
x_train = data["trainx"]
y_train = data["trainy"]

training_number = x_train.shape[0]
test_number = x_test.shape[0]

from sklearn.ensemble import RandomForestRegressor
regr = RandomForestRegressor(n_estimators=20, max_depth=12, random_state=0)

regr.fit(x_train, y_train)

y_predicted = regr.predict(x_test)

index_max = np.where(y_predicted > max(y_train))
index_min = np.where(y_predicted < min(y_train))
y_predicted[index_max] = max(y_train)
y_predicted[index_min] = min(y_train)

predict = y_predicted.astype(int)

import csv
writer = csv.writer(open("/Users/cee/Desktop/prediction_test1.csv", "w"))
writer.writerow(["dataid", "prediction"])
for index, value in enumerate(predict):
    writer.writerow([index + 1, value])