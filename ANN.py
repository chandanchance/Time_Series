# -*- coding: utf-8 -*-
"""
Created on Mon Oct  1 10:50:55 2018

@author: Chandan
"""
# libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

considerLast = 10
neurons = 50
trainTestCutOffAt = 90

# Importing the dataset

dataset = pd.read_csv(r'Somnath.csv')

dataset_train = dataset[0: trainTestCutOffAt]
dataset_test = dataset[trainTestCutOffAt-considerLast:]

training_set = dataset_train.iloc[:, 1:2].values
testing_set = dataset_test.iloc[:, 1:2].values

# Feature Scaling
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range = (0, 1))
training_set_scaled = sc.fit_transform(training_set)
testing_set_scaled = sc.transform(testing_set)

# Creating a data structure with considerLast timesteps
X_train = []
y_train = []
for i in range(considerLast, len(training_set_scaled)):
    X_train.append(training_set_scaled[i-considerLast:i, 0])
    y_train.append(training_set_scaled[i, 0])
X_train, y_train = np.array(X_train), np.array(y_train)


X_test = []
y_test = []
for i in range(considerLast, len(testing_set_scaled)):
    X_test.append(testing_set_scaled[i-considerLast:i, 0])
    y_test.append(testing_set_scaled[i, 0])
X_test, y_test = np.array(X_test), np.array(y_test)



# Importing the Keras libraries and packages
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout

# Initialising the ANN
Regression = Sequential()

# Adding the input layer and the first hidden layer
Regression.add(Dense(units = neurons, activation = 'relu', input_dim = (X_train.shape[1])))

# Adding the hidden layers
Regression.add(Dense(units = neurons, activation = 'relu'))
Regression.add(Dropout(p = 0.2))

Regression.add(Dense(units = neurons, activation = 'relu'))
Regression.add(Dropout(p = 0.2))

Regression.add(Dense(units = neurons, activation = 'relu'))
Regression.add(Dropout(p = 0.2))

Regression.add(Dense(units = 1, activation = 'relu'))

# Compiling the ANN
Regression.compile(optimizer = 'adam', loss = 'mean_absolute_error', metrics = ['accuracy'])

# Fitting the ANN to the Training set
Regression.fit(X_train, y_train, batch_size = considerLast, epochs = 100)

#Making predictions and evaluating the model
# Predicting the Test set results
predicted_value = Regression.predict(X_test)
predicted_value = sc.inverse_transform(predicted_value)


plt.plot(testing_set[considerLast:], color = 'red', label = 'Actual')
plt.plot(predicted_value, color = 'blue', label = 'Predicted')
plt.title('Somnath')
plt.xlabel('Time')
plt.ylabel('Value')
plt.legend()
plt.show()

from sklearn.metrics import mean_absolute_error
print("MAE")
print(mean_absolute_error(testing_set[considerLast:],predicted_value))
print("Accuraccy")
accuracy = 1-np.mean(abs(testing_set[considerLast:] - predicted_value)/testing_set[considerLast:])
print('accuracy is {0:.2f}%'.format(accuracy*100))


def mean_absolute_percentage_error(y_true, y_pred): 
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

mape = mean_absolute_percentage_error(testing_set[considerLast:],predicted_value)
print('Test mape: %.2f' % mape)

print("Accuracy : ", 100-mape)

def one_Step_forecast(model,X,sc):
    values = model.predict(X)
    values = sc.inverse_transform(values)
    return(values[-1])


def forecast(model,data,noOfForecasts):
    forecasts = []
    originalData = data
    
    for i in range(0,noOfForecasts):
        data = np.append(data,[np.mean(data)])
        data = data.reshape(-1,1)
        sc = MinMaxScaler(feature_range = (0, 1))
        dataScaled = sc.fit_transform(data)
        
        X = []
        y = []
        for i in range(considerLast, len(dataScaled)):
            X.append(dataScaled[i-considerLast:i, 0])
            y.append(dataScaled[i, 0])
        X, y= np.array(X), np.array(y)
        
        
        nextForecast = one_Step_forecast(model,X,sc)
        forecasts.append(nextForecast)
        data = np.append(data[:-1],nextForecast)
        data = data
    
    return forecasts
    
a = forecast(Regression,dataset_train.iloc[:, 1:2].values,5)
    
    
    
#Code for forecasting the values
    
forecast = []
data =dataset_train.iloc[:, 1:2].values
originalData = data
data = np.append(data,[np.mean(data)])
data = data.reshape(-1,1)
sc = MinMaxScaler(feature_range = (0, 1))
dataScaled = sc.fit_transform(data)
    
X = []
y = []
for i in range(considerLast, len(dataScaled)):
    X.append(dataScaled[i-considerLast:i, 0])
    y.append(dataScaled[i, 0])
X, y= np.array(X), np.array(y)
    
values = Regression.predict(X)
values = sc.inverse_transform(values)
nextForecast=values[-1]
forecast.append(nextForecast)
data = np.append(data[:-1],nextForecast)
data = data
    
