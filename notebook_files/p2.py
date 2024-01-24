
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report,accuracy_score
from sklearn.preprocessing import MinMaxScaler
import pickle


diabetes_df = pd.read_csv('diabetes_dataset.csv')


cols = ['SkinThickness','Outcome']
features = diabetes_df.drop(cols,axis='columns')
target = diabetes_df['Outcome']

mms = MinMaxScaler() 
sfeatures = mms.fit_transform(features)
print(sfeatures)

x_train,x_test,y_train,y_test = train_test_split(sfeatures,target,random_state = 2)

model = RandomForestClassifier()
model.fit(x_train,y_train)

y_pred = model.predict(x_test)
cr = classification_report(y_test,y_pred);
print(cr)
print("Accuracy = ", accuracy_score(y_test,y_pred))

with open('diabetes_c.pkl', 'wb') as model_file:
    pickle.dump(model, model_file, protocol=4)
with open('scalerdiab.pkl', 'wb') as scaler_file:
    pickle.dump(mms, scaler_file, protocol=4)

with open('diabetes_c.pkl', 'rb') as model_file:
    cb = pickle.load(model_file)