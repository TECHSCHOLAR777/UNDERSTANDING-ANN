import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import MeanSquaredError
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

X = np.linspace(-1000, 1000, 10000).reshape(-1, 1)
y = X ** 2

X = X / 1000
y = y / 1e6

noise = np.random.normal(0, 0.03, size=y.shape)
y_noisy = y + noise

X_train, X_test, y_train, y_test = train_test_split(
    X, y_noisy, test_size=0.2, random_state=777
)

X_mean, X_std = X_train.mean(), X_train.std()
y_mean, y_std = y_train.mean(), y_train.std()

X_train = (X_train - X_mean) / X_std
X_test = (X_test - X_mean) / X_std

y_train = (y_train - y_mean) / y_std
y_test = (y_test - y_mean) / y_std

model = Sequential([
    Dense(64, activation='tanh', input_shape=(1,)),
    Dense(32, activation='tanh'),
    Dense(16, activation='tanh'),
    Dense(1)
])

model.compile(
    optimizer=Adam(learning_rate=0.0008),
    loss=MeanSquaredError()
)

history = model.fit(
    X_train, y_train,
    epochs=100,
    batch_size=32,
    validation_data=(X_test, y_test)
)

plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

predictions = model.predict(X_test)

X_plot = X_test * X_std + X_mean
y_plot = y_test * y_std + y_mean
pred_plot = predictions * y_std + y_mean

plt.scatter(X_plot*1000, y_plot*1000000, label='Noisy Data', alpha=0.3)
plt.scatter(X_plot  *1000, pred_plot*1000000, color='red', label='Model Predictions', alpha=0.5)
plt.xlabel('X')
plt.ylabel('y')
plt.legend()
plt.show()

print("Sample X (original scale):")
print((X_plot[:5] * 1000).flatten())

print("\nTrue y (original scale):")
print((y_plot[:5] * 1e6).flatten())

print("\nPredicted y (original scale):")
print((pred_plot[:5] * 1e6).flatten())
