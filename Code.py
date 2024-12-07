import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.optimizers import Adam

file_path = 'Electric_Production.csv'
data = pd.read_csv(file_path)
data['DATE'] = pd.to_datetime(data['DATE'], format='%m/%d/%Y')
data.set_index('DATE', inplace=True)
scaler = MinMaxScaler(feature_range=(0, 1))
data['IPG2211A2N'] = scaler.fit_transform(data[['IPG2211A2N']])

def create_dataset(data, time_step=1):
    X, Y = [], []
    for i in range(len(data) - time_step):
        X.append(data[i:i + time_step, :])
        Y.append(data[i + time_step, 0])
    return np.array(X), np.array(Y)

time_step = 10
data_values = data[['IPG2211A2N']].values
X, Y = create_dataset(data_values, time_step)
X = X.reshape(X.shape[0], X.shape[1], 1)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

univariate_model = Sequential([
    LSTM(50, return_sequences=False, input_shape=(time_step, 1)),
    Dense(1)
])

univariate_model.compile(optimizer=Adam(learning_rate=0.01), loss='mean_squared_error')
univariate_model.fit(X_train, Y_train, epochs=20, batch_size=32, validation_data=(X_test, Y_test), verbose=1)

data['Lag_1'] = data['IPG2211A2N'].shift(1)
data['Lag_2'] = data['IPG2211A2N'].shift(2)
data.dropna(inplace=True)

multivariate_data = data[['IPG2211A2N', 'Lag_1', 'Lag_2']].values
X_multi, Y_multi = create_dataset(multivariate_data, time_step)
X_multi = X_multi.reshape(X_multi.shape[0], X_multi.shape[1], 3)
X_train_multi, X_test_multi, Y_train_multi, Y_test_multi = train_test_split(X_multi, Y_multi, test_size=0.2, random_state=42)

multivariate_model = Sequential([
    LSTM(50, return_sequences=False, input_shape=(time_step, 3)),
    Dense(1)
])

multivariate_model.compile(optimizer=Adam(learning_rate=0.01), loss='mean_squared_error')
multivariate_model.fit(X_train_multi, Y_train_multi, epochs=20, batch_size=32, validation_data=(X_test_multi, Y_test_multi), verbose=1)

univariate_loss = univariate_model.evaluate(X_test, Y_test, verbose=0)
multivariate_loss = multivariate_model.evaluate(X_test_multi, Y_test_multi, verbose=0)

print(f"Univariate Model Loss: {univariate_loss}")
print(f"Multivariate Model Loss: {multivariate_loss}")
