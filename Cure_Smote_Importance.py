import numpy as np
import pandas as pd
from sklearn import svm
import smote_variants as sv
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold


#data_file = ['transfusion.csv', 'haberman.csv', 'breast-cancer-wisconsin.csv']
data_file = ['transfusion.csv', 'haberman.csv', 'anova-result.csv']
file_no = 2
fold_numbers = 5
print(data_file[file_no])
# loading the dataset
trans_data = pd.read_csv(data_file[file_no]).to_numpy()
X_data, Y_data = trans_data[:, 0:-1], trans_data[:, -1]
feature_names = (pd.DataFrame(trans_data)).head()
# print(X_data.shape, Y_data.shape)
# standardizing data and declaring k-fold cross validation
scaler = StandardScaler()
X_data = scaler.fit_transform(X_data)
skf = StratifiedKFold(n_splits=fold_numbers, shuffle=True, random_state=42)

g = 1
outputs = [[]]
index = 0

# using CURE_SMOTE oversampling
print("-----------using CURE_SMOTE--------------")
oversampler2 = sv.CURE_SMOTE()
kernels = ['linear', 'poly', 'rbf', 'sigmoid']
for krl in kernels:
    accuracy, precision, recall, f1 = 0.0, 0.0, 0.0, 0.0
    for train_index, test_index in skf.split(X_data, Y_data):
        X_train, X_test = X_data[train_index], X_data[test_index]
        y_train, y_test = Y_data[train_index], Y_data[test_index]
        X_samp, y_samp = oversampler2.sample(X_train, y_train)
        clf = svm.SVC(kernel=krl, random_state=1, gamma=0.1, C=0.1)  # using RBF kernel
        clf.fit(X_samp, y_samp)
        y_pred = clf.predict(X_test)
        precision = precision + metrics.precision_score(y_test, y_pred)
        recall = recall + metrics.recall_score(y_test, y_pred)
        f1 = f1 + metrics.f1_score(y_test, y_pred)
        accuracy = accuracy + metrics.accuracy_score(y_test, y_pred)
    precision, recall, f1, accuracy = precision / 5.0, recall / 5.0, f1 / 5.0, accuracy/5.0
    print(krl)
    print("Precision:", precision)
    print("Recall:", recall)
    print("F1 Score:", f1)
    print("Accuracy:", accuracy)
    outputs.append([krl, precision, recall, f1])
    print("----------------------------------")
