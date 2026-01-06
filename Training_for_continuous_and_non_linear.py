import numpy as np
from Standard_NN_Training_Function import train_neural_network, plot_loss_curves, plot_predictions, sample_predictions

# Data generation
X = np.linspace(-1000, 1000, 10000).reshape(-1, 1)                 
y = X ** 2                   # quadratic function (already well-scaled)



# Optional small noise (can keep or remove)
noise = np.random.normal(0, 0.02, size=y.shape)
y_noisy = y + noise




# Example usage of the modular functions
model, history, maxx, maxy, X_val, y_val, valid_predictions = train_neural_network(
    X, y_noisy,
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

