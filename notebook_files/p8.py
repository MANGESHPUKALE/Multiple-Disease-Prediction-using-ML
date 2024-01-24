import pandas as pd
import numpy as np
import pickle


data = pd.read_csv('lung.csv')
data.head()

one_hot = pd.get_dummies(data['GENDER'])
data = data.drop('GENDER',axis = 1)
data = data.join(one_hot)
data.head()

from sklearn import preprocessing

label = preprocessing.LabelEncoder()
data['LUNG_CANCER'] = label.fit_transform(data['LUNG_CANCER'])
data.head()


cols = ['YELLOW_FINGERS', 'PEER_PRESSURE', 'WHEEZING','LUNG_CANCER']
x = data.drop(cols, axis=1)
y = data['LUNG_CANCER']

from sklearn.model_selection import train_test_split

train_x,test_x,train_y,test_y = train_test_split(x,y)

from sklearn.svm import SVC

model = SVC(kernel='rbf', C=1.0, gamma='scale')
model.fit(train_x,train_y)

# Make predictions on the test set
y_pred = model.predict(test_x)

# Calculate accuracy
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(test_y, y_pred)
print("Accuracy =", accuracy)

with open('lung_c.pkl', 'wb') as model_file:
    pickle.dump(model, model_file, protocol=4)
with open('scalerlung.pkl', 'wb') as label_file:
    pickle.dump(label, label_file, protocol=4)

with open('lung_c.pkl', 'rb') as model_file:
    cb = pickle.load(model_file)

