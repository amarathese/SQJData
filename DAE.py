# -*- coding: utf-8 -*-
"""
Created on Fri Jun  2 10:30:09 2023

@author: Amara Dalila
"""

import numpy as np
from sklearn import svm
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model
import pandas as pd

# Load and preprocess your dataset
# Assuming X_train and y_train are your training data and labels

df = pd.read_csv('Data/xerces-1.3.csv')
#df=pd.from_csv('Data/churn.csv', index_col=None);
#df.columns = df.columns.str.strip()
 

#Step 3: Separating the dependent 'bugs' and independent variables 'metrics'




# Separating the normal and fraudulent transactions

target=df[df.columns[-1]]


fraud = target!= 0
normal = target== 0

 
# Separating the dependent and independent variables
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler 
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns
from keras.layers import Input, Dense
from keras.models import Model, Sequential
from keras import regularizers
import numpy as np

df[df.columns[-1]]=df[df.columns[-1]].apply(lambda x: 1 if x!=0 else 0)
 


y=df[df.columns[-1]]
X = df.drop(df.columns[-1], axis = 1)
X = df.drop(df.columns[0:3], axis = 1)

print("xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx", X.shape)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the data
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Add noise to the training data
noise_factor = 0.1
X_train_noisy = X_train_scaled + noise_factor * np.random.normal(size=X_train_scaled.shape)

# Define the autoencoder architecture
input_dim = X_train_noisy.shape[1]
print("input shaaaaaaaaaaaape",input_dim )
encoding_dim = 10

input_layer = Input(shape=(input_dim,))
encoded = Dense(encoding_dim, activation='relu')(input_layer)
decoded = Dense(input_dim, activation='sigmoid')(encoded)

autoencoder = Model(input_layer, decoded)

# Compile and train the autoencoder
autoencoder.compile(optimizer='adam', loss='mean_squared_error')
autoencoder.fit(X_train_noisy, X_train_scaled, epochs=100, batch_size=16, shuffle=True, validation_data=(X_test_scaled, X_test_scaled))




import time
start = time.time()
DAE_history=autoencoder.fit(X_train_noisy, X_train_noisy, 
                batch_size = 16, epochs = 100, 
                shuffle = True, validation_split = 0.20)
stop = time.time()
print(f"Training time: {stop - start}s")



# Get the learned features from the encoder part of the autoencoder
encoder = Model(input_layer, encoded)
encoded_X_train = encoder.predict(X_train_scaled)
encoded_X_test = encoder.predict(X_test_scaled)

# Train an SVM classifier on the encoded features
svm_classifier = svm.SVC()
svm_classifier.fit(encoded_X_train, y_train)

# Make predictions on the test set using the SVM classifier
svm_predictions = svm_classifier.predict(encoded_X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, svm_predictions)
print("Accuracy:", accuracy)
