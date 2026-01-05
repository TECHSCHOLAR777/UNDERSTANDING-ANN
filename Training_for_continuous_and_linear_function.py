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

 # choosing function y=10x+43

X=np.linspace(-1000,1000,10000).reshape(-1,1)
##############################
y=5*X+4
X=X/1000  #normalizing the input data
#now we area adding some noise to the data
noise=0
noisyy=y+noise
noisyy=noisyy.astype(np.float32)
noisyy=noisyy/1000  #normalizing the output data    

Model=Sequential([
    Dense(8,activation='relu',input_shape=(1,)),
    Dense(4,activation='relu'),
    Dense(1)
]) 
Model.compile(optimizer=Adam(learning_rate=0.001),loss=MeanSquaredError())
X_train,X_test,y_train,y_test=train_test_split(X,noisyy,test_size=0.2,random_state=777)
history=Model.fit(X_train,y_train,validation_data=(X_test,y_test),epochs=50,batch_size=32)
plt.plot(history.history['loss'],label='Training Loss')
plt.plot(history.history['val_loss'],label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()
predictions=Model.predict(X_test)
plt.scatter(X_test,y_test,label='Noisy Data',alpha=0.3)
plt.scatter(X_test,predictions,color='r',label='Model Predictions',alpha=0.5)
plt.xlabel('X')
plt.ylabel('y')
plt.legend()
plt.show()
print("")
print(X_test[:5])
print(predictions[:5])