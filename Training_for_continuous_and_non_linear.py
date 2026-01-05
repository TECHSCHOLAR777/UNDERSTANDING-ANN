import numpy as np 
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential      
from tensorflow.keras.layers import Dense   
from tensorflow.keras.optimizers import Adam    
from sklearn.model_selection import train_test_split    
from tensorflow.keras.losses import MeanSquaredError


import matplotlib.pyplot as plt

X = np.linspace(-1000, 1000, 10000).reshape(-1,1)
y = X**2

# normalize
X = X / 1000
y = y / 1e6

# small noise
noise = np.random.normal(0, 0.03, size=y.shape)
y_noisy = y + noise

X_train, X_test, y_train, y_test = train_test_split(X, y_noisy, test_size=0.2, random_state=777) 
model = Sequential([
    Dense(32, activation='tanh', input_shape=(1,)),
    Dense(1)
])

model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss=MeanSquaredError()
)

history = model.fit(
    X_train, y_train,
    epochs=100,
    batch_size=32,
    validation_data=(X_test, y_test)            
)
plt.plot(history.history['loss'],label='Training Loss')
plt.plot(history.history['val_loss'],label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()
predictions=model.predict(X_test)
plt.scatter(X_test,y_test,label='Noisy Data',alpha=0.3)
plt.scatter(X_test,predictions,color='r',label='Model Predictions',alpha=0.5)
plt.xlabel('X')
plt.ylabel('y')
plt.legend()
plt.show()
print("")
print(X_test[:5])
print(y_test[:5])