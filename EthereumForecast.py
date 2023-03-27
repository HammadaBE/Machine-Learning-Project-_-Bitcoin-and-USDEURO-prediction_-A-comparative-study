import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM

# Load the Ethereum price data
df = pd.read_csv('coin_Ethereum - Trimmed.csv')

# Split the data into training and testing sets
X = df[['Open', 'High', 'Low', 'Volume']]
y = df['Close']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Scale the data
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Reshape the data for LSTM input
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

# Build the LSTM model
model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
model.add(LSTM(units=50))
model.add(Dense(units=1))
model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
model.fit(X_train, y_train, epochs=100, batch_size=32)

# Evaluate the model on the test set
mse = model.evaluate(X_test, y_test)
print(f'Test set MSE: {mse:.4f}')

# Make predictions on new data
new_data = np.array([[3000, 3500, 2800, 5000000]])
new_data = sc.transform(new_data)
new_data = np.reshape(new_data, (new_data.shape[0], new_data.shape[1], 1))
prediction = model.predict(new_data)
print(f'Predicted Ethereum price: {prediction[0][0]:.2f}')
