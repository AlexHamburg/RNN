# -*- coding: utf-8 -*-
"""
Created on Thu Oct 11 11:12:45 2018

@author: Oleksandr Trunov
Recurrent Neural Network
"""
import numpy as np
import pandas as pd
import seaborn as sbn
import random
import matplotlib.pyplot as plt
# Part 1 - Data Preprocessing
# Step 1.1 - Importing of data
default_seed = 2
random.seed(default_seed)
train_csv = pd.read_csv("Google_Stock_Price_Train.csv")
# Add .values to make as np.array
training_set = train_csv.iloc[:,1].values
# Check the data
print(np.isnan(training_set).any())
# Visualisation
sbn.set(context="notebook", style="dark", palette="muted")
sbn.lineplot(data=training_set)
# Step 1.2 - Feature scaling (Normalisation and Standardisation)
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0,1))
training_set_scaled = scaler.fit_transform(training_set.reshape(-1,1))
# Choose a right number of timesteps for 1 output
# 2D Array with 60 previous values
X_train = []
# 1D Array with value of 60th timestep
Y_train = []
# test of 60 timesteps
for x in range(60, len(training_set)):
    X_train.append(training_set_scaled[x-60:x, 0])
    Y_train.append(training_set_scaled[x, 0])
X_train, Y_train = np.array(X_train), np.array(Y_train)
# Reshaping of Data with some more idicators, adding of a new dimension
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
# Part 2 - RNN Building
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout
from keras.callbacks import ReduceLROnPlateau, CSVLogger
# Initialising the RNN
rnn = Sequential()
# LSTM Layers
# param return_sequence - if you have more than 1 LSTM-layer
rnn.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
rnn.add(Dropout(0.2))
rnn.add(LSTM(units=50, return_sequences=True))
rnn.add(Dropout(0.2))
rnn.add(LSTM(units=50, return_sequences=True))
rnn.add(Dropout(0.2))
rnn.add(LSTM(units=50))
rnn.add(Dropout(0.2))
# Full connected ANN (output layer)
rnn.add(Dense(units=1))

rnn.compile(optimizer="adam", loss="mean_squared_error")
# Callback
learning_rate_callback = ReduceLROnPlateau(monitor="val_loss", factor=0.2, patience=3)
csv_logger = CSVLogger("rnn_ot.csv")
# Fitting with data
rnn.fit(x=X_train, y=Y_train, batch_size=32, epochs=100, callbacks=[learning_rate_callback, csv_logger])
# Visualisation of loss
csv_rnn_loss = pd.read_csv("rnn_ot.csv")
sbn.lineplot(x="epoch", y="loss", data=csv_rnn_loss)
# Part 3 - Prediction
test_csv = pd.read_csv("Google_Stock_Price_Test.csv")
test_set = test_csv.iloc[:,1].values

dataset_total = pd.concat((train_csv['Open'], test_csv['Open']), axis=0)
inputs = dataset_total[len(dataset_total)-len(test_set)-60:].values
inputs = inputs.reshape(-1,1)
# Reskale the input
inputs = scaler.transform(inputs)
# 3D Array for Keras
X_test = []
for x in range(60, 80):
    X_test.append(inputs[x-60:x, 0])
X_test = np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
prediction = rnn.predict(X_test)
prediction = scaler.inverse_transform(prediction)
# Visualisation
plt.plot(test_set, color='red', label="Real")
plt.plot(prediction, color="blue", label="Predict")
plt.title("Google Price")
plt.xlabel("Date")
plt.ylabel("Price")
plt.legend()
plt.show()
# Part 4 - Evaluation