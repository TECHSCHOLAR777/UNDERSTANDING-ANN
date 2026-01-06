import numpy as np
import tensorflow as tf 
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import MeanSquaredError
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

def relative_mse(y_true, y_pred):
    eps = 1e-6
    return tf.reduce_mean(
        tf.square(y_pred - y_true) / (tf.abs(y_true) + eps)
    )


def train_neural_network(X, y, layers, units, activations, learning_rate=0.0005, epochs=75, batch_size=32,scaling=True,test_ratio=0.2):
    model = Sequential()
    for i in range(layers):
        if i == 0:
            model.add(Dense(units[i], activation=activations[i], input_shape=(X.shape[1],)))
        else:
            model.add(Dense(units[i], activation=activations[i]))
    
    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss=relative_mse # custom loss function, hum kabhi call nhi karenge directly tensorflow handle karega
    )
    
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=test_ratio, random_state=777)

    if scaling:
        maxx = X_train.max()
        maxy = np.max(np.abs(y_train))
        X_train = X_train / maxx
        X_val = X_val / maxx
        y_train = y_train / maxy
        y_val = y_val / maxy
    else:
        maxx = 1
        maxy = 1




    history = model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(X_val, y_val)
    )

    valid_predictions = model.predict(X_val)
    
    return model, history, maxx, maxy, X_val, y_val, valid_predictions

def plot_loss_curves(history):
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()
def plot_predictions(X, y_true, y_pred, maxx=1, maxy=1):
    plt.scatter(X * maxx, y_true * maxy, alpha=0.3, label='True (noisy)')
    plt.scatter(X * maxx, y_pred * maxy, color='red', alpha=0.5, label='Predicted')
    plt.xlabel('X (original scale)')
    plt.ylabel('y')
    plt.legend()
    plt.show()
def sample_predictions(X, y_true, y_pred, maxx=1, maxy=1, n=5):
    print("Sample X (original scale):")
    print((X[:n] * maxx))

    print("\nTrue y:")
    print(y_true[:n] * maxy)

    print("\nPredicted y:")
    print(y_pred[:n] * maxy)
