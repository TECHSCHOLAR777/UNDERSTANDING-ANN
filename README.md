🤖 ANN Function Approximation (Exploration)
📘 About this Project
This project is an exploration of how Artificial Neural Networks (ANNs) learn and approximate different mathematical functions.
🎯 The aim was not to get perfect predictions, but to understand:
- 🔍 How neural networks behave with different types of functions
- ⚡ How activation functions affect learning
- 📏 Why scaling and noise matter
- 🚧 Where neural networks struggle
This project was done purely as a learning exercise 🧑‍💻.

🧩 Functions Explored
I experimented with different types of functions, increasing complexity step by step:
- 📈 Linear function → y = 2x + 3
- 🔄 Non-linear function → y = x²
- ✂️ Non-differentiable function → y = x
- ⏹️ Discontinuous function → Step function
- 🎲 Discrete function → Floor function & XOR
- ➕ Multi-variable function → y = x₁ + x₂
- 🌊 Complex function → y = sin(x₁) + x₂²

🛠️ Dataset Creation
- 📊 Input data generated using NumPy
- 🧮 Outputs calculated using mathematical expressions
- 🎛️ Small noise added to simulate real-world data
- 🔄 Both input and output normalized to improve training
💡 Example:
X = np.linspace(-1000, 1000, 10000).reshape(-1, 1)
y = X**2

X = X / 1000
y = y / 1e6

noise = np.random.normal(0, 0.02, size=y.shape)
y_noisy = y + noise



🧠 Neural Network Models
- Built using TensorFlow (Keras) ⚙️
- Mostly simple architectures 🏗️
- Tried different activation functions:
- 🔥 ReLU
- 🌈 Tanh
- 📉 Loss function: Mean Squared Error (MSE)
- 🚀 Optimizer: Adam
💡 Example model:
model = Sequential([
    Dense(32, activation='tanh', input_shape=(1,)),
    Dense(1)
])



👀 Observations
Some interesting things I noticed:
- ❌ Linear models fail on non-linear functions
- 📐 ReLU networks approximate curves using straight-line segments
- 🔀 For y = x², the model sometimes learned a V-shape (x) instead of a parabola
- ✅ Normalizing both input and output greatly improved learning
- ⚠️ Large learning rates caused unstable training
- 🌪️ Too much noise made learning difficult

🏁 Conclusion
This task helped me understand that:
- 🧠 Neural networks approximate functions rather than learning exact formulas
- 📏 Data scaling and activation functions play a major role
- 🏗️ “Good architecture” alone does not guarantee good learning
✨ Overall, this project improved my intuition about how ANNs actually work.


