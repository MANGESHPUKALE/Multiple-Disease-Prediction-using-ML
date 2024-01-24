import matplotlib.pyplot as plt
from sklearn.svm import SVC  # Import the Support Vector Machine classifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import MinMaxScaler
import pickle
import pandas as pd
import numpy as np

heart_df = pd.read_csv('heart.csv')

cols = ['target']
features = heart_df.drop(cols, axis=1)
target = heart_df['target']

mms = MinMaxScaler()
sfeatures = mms.fit_transform(features)
print(sfeatures)

x_train, x_test, y_train, y_test = train_test_split(sfeatures, target, random_state=2)

# Create and train the SVM model
model = SVC(kernel='rbf')  
model.fit(x_train, y_train)

y_pred = model.predict(x_test)
cr = classification_report(y_test, y_pred)
print(cr)
print("Accuracy =", accuracy_score(y_test, y_pred))

with open('heart_c.pkl', 'wb') as model_file:
    pickle.dump(model, model_file, protocol=4)
with open('scalerheart.pkl', 'wb') as scaler_file:
    pickle.dump(mms, scaler_file, protocol=4)

with open('heart_c.pkl', 'rb') as model_file:
    cb = pickle.load(model_file)


