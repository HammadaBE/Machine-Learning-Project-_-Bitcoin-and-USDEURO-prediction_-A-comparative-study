import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler

# Load the data
df = pd.read_csv('bitcoin_data.csv')

# Normalize the data
scaler = MinMaxScaler(feature_range=(0, 1))
df_scaled = scaler.fit_transform(df[['Open_price', 'Low_price', 'High_price', 'Close_price']])

# Split the data into training and testing sets
train_size = int(len(df_scaled) * 0.8)
test_size = len(df_scaled) - train_size
train_data, test_data = df_scaled[0:train_size, :], df_scaled[train_size:len(df_scaled), :]

# Define the function to create the input and output sequences
def create_sequences(dataset, seq_length):
    data_X, data_y = [], []
    for i in range(len(dataset)-seq_length-1):
        a = dataset[i:(i+seq_length), :]
        data_X.append(a)
        data_y.append(dataset[i + seq_length, 3])
    return np.array(data_X), np.array(data_y)

# Set the sequence length
seq_length = 30

# Create the input and output sequences
train_X, train_y = create_sequences(train_data, seq_length)
test_X, test_y = create_sequences(test_data, seq_length)

# Define the LSTM model
model = tf.keras.Sequential([
    tf.keras.layers.LSTM(50, return_sequences=True, input_shape=(train_X.shape[1], train_X.shape[2])),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.LSTM(100, return_sequences=True),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.LSTM(100),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(1)
])

# Compile the model
model.compile(loss='mean_squared_error', optimizer='adam')

# Train the model
model.fit(train_X, train_y, epochs=100, batch_size=64, validation_split=0.1, verbose=1)

# Evaluate the model
train_score = model.evaluate(train_X, train_y, verbose=0)
print('Train loss:', train_score)
test_score = model.evaluate(test_X, test_y, verbose=0)
print('Test loss:', test_score)

# Make predictions
train_predict = model.predict(train_X)
test_predict = model.predict(test_X)

# Inverse the scaling of the predictions
train_predict = scaler.inverse_transform(np.concatenate((train_X[:,:,0], train_predict), axis=1))[:, -1]
test_predict = scaler.inverse_transform(np.concatenate((test_X[:,:,0], test_predict), axis=1))[:, -1]
train_y = scaler.inverse_transform(np.concatenate((train_X[:,:,0], train_y.reshape(-1,1)), axis=1))[:, -1]
test_y = scaler.inverse_transform(np.concatenate((test_X[:,:,0], test_y.reshape(-1,1)), axis=1))[:, -1]

# Plot the predictions
import matplotlib.pyplot as plt
plt.plot(train_y)
plt.plot(train_predict)
plt.plot(test_y)
plt.plot(test_predict)
plt.legend(['Train Close', 'Predicted Train Close', 'Test Close', 'Predicted Test Close'])
plt.show()
