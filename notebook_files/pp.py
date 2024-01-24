import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import MinMaxScaler
import pickle
import numpy as np

# Load the dataset
parkinsons_df = pd.read_csv('parkinsons.csv')

# Define the columns to drop
cols_to_drop = ['name','MDVP:Jitter(%)','MDVP:Jitter(Abs)','MDVP:Shimmer(dB)','Shimmer:APQ3','NHR','HNR','RPDE','DFA','Shimmer:DDA','Shimmer:APQ5','MDVP:RAP','status']
features = parkinsons_df.drop(cols_to_drop, axis='columns')
target = parkinsons_df['status']

# Scale the features
mms = MinMaxScaler()
sfeatures = mms.fit_transform(features)

# Split the data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(sfeatures, target, random_state=134)

# Create a Random Forest Classifier
model = RandomForestClassifier(n_estimators=20)

# Train the model
model.fit(x_train, y_train)

# Make predictions
y_pred = model.predict(x_test)

# Evaluate the model
cr = classification_report(y_test, y_pred)
accuracy = accuracy_score(y_test, y_pred)
print(cr)
print("Accuracy =", accuracy)

with open('parkinson_c.pkl', 'wb') as model_file:
    pickle.dump(model, model_file, protocol=4)
with open('scalerpark.pkl', 'wb') as scaler_file:
    pickle.dump(mms, scaler_file, protocol=4)

with open('parkinson_c.pkl', 'rb') as model_file:
    cb = pickle.load(model_file)