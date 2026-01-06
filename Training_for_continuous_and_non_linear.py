import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import MeanSquaredError
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Data generation
X = np.linspace(-1000, 1000, 10000).reshape(-1, 1)
X = X / 1000                 # scale to [-1, 1]
y = X ** 3                   # cubic function (already well-scaled)

# Optional small noise (can keep or remove)
noise = np.random.normal(0, 0.02, size=y.shape)
y_noisy = y + noise

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y_noisy, test_size=0.2, random_state=777
)

# Z-score normalization (train statistics only)
X_mean, X_std = X_train.mean(), X_train.std()
y_mean, y_std = y_train.mean(), y_train.std()

X_train = (X_train - X_mean) / X_std
X_test  = (X_test  - X_mean) / X_std

y_train = (y_train - y_mean) / y_std
y_test  = (y_test  - y_mean) / y_std

# Model
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

# Training
history = model.fit(
    X_train, y_train,
    epochs=50,
    batch_size=64,
    validation_data=(X_test, y_test)
)

# Loss curves
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

# Predictions
predictions = model.predict(X_test)

# De-normalize
X_plot = X_test * X_std + X_mean
y_plot = y_test * y_std + y_mean
pred_plot = predictions * y_std + y_mean

# Plot in ORIGINAL SCALE
plt.scatter(X_plot * 1000, y_plot, alpha=0.3, label='True (noisy)')
plt.scatter(X_plot * 1000, pred_plot, color='red', alpha=0.5, label='Predicted')
plt.xlabel('X (original scale)')
plt.ylabel('y')
plt.legend()
plt.show()

# Sample values
print("Sample X (original scale):")
print((X_plot[:5] * 1000).flatten())

print("\nTrue y:")
print(y_plot[:5].flatten())

print("\nPredicted y:")
print(pred_plot[:5].flatten())
