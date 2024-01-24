import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.preprocessing import MinMaxScaler
import pickle

from sklearn.datasets import load_breast_cancer
cancer_dataset = load_breast_cancer()
print(cancer_dataset)

print(cancer_dataset['filename'])

cancer_df = pd.DataFrame(np.c_[cancer_dataset['data'],cancer_dataset['target']],columns=np.append(cancer_dataset['feature_names'],['target']))

print(cancer_df.head())


cols = ['mean radius','mean area','mean smoothness','mean concavity','mean concave points','mean symmetry','mean fractal dimension','radius error','area error','compactness error','concavity error','concave points error','fractal dimension error','worst radius','worst smoothness','worst concavity','worst concave points','worst symmetry','worst fractal dimension','target']
features = cancer_df.drop(cols,axis='columns')
target = cancer_df['target']

mms = MinMaxScaler() 
sfeatures = mms.fit_transform(features)
print(sfeatures)

x_train,x_test,y_train,y_test= train_test_split(sfeatures,target,random_state=134)

model = RandomForestClassifier(n_estimators=20)
model.fit(x_train,y_train)

y_pred = model.predict(x_test)
cr = classification_report(y_test,y_pred)
print(cr)


with open('breast_c.pkl', 'wb') as model_file:
    pickle.dump(model, model_file, protocol=4)
with open('scaler.pkl', 'wb') as scaler_file:
    pickle.dump(mms, scaler_file, protocol=4)

with open('breast_c.pkl', 'rb') as model_file:
    cb = pickle.load(model_file)


fa = float(input("Enter Mean Texture "))
fb = float(input("Enter Mean Perimeter "))
fc = float(input("Enter Mean Compactness "))
fd = float(input("Enter Texture Error "))
fe = float(input("Enter Perimeter Error "))
ff = float(input("Enter Smoothness Error "))
fg = float(input("Enter Symmetry Error "))
fh = float(input("Enter Worst Texture "))
fi = float(input("Enter Worst Perimeter "))
fj = float(input("Enter Worst Area "))
fk = float(input("Enter Worst Compactness "))
d = [[fa,fb,fc,fd,fe,ff,fg,fh,fi,fj,fk]]
new_d = mms.transform(d)
res = model.predict(new_d)
print(res)