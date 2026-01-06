import numpy as np 
from Standard_NN_Training_Function import train_neural_network, plot_loss_curves, plot_predictions, sample_predictions


 # choosing function y=10x+43

X=np.linspace(-10000,10000,10000).reshape(-1,1)
##############################
y=5*X+4

noise=np.random.normal(0, 0.02, size=y.shape)
noisyy=y+noise

model, history, maxx, maxy, X_val, y_val, valid_predictions=train_neural_network(
    X, noisyy,
    layers=4,
    units=[64, 32, 16, 1],
    activations=['relu', 'relu', 'relu', None],
    learning_rate=0.0005,
    epochs=75,
    batch_size=64,
    scaling=True,
    test_ratio=0.2
)   

plot_loss_curves(history)
plot_predictions(X_val, y_val, valid_predictions, maxx, maxy)
sample_predictions(X_val, y_val, valid_predictions, maxx, maxy, n=5)
print("Model Summary:")
model.summary()
