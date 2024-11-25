import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Circle
import os
from functools import partial
import matplotlib
matplotlib.use('Agg')

result_dir = "results"
os.makedirs(result_dir, exist_ok=True)

# Define a simple MLP class
class MLP:
    def __init__(self, input_dim, hidden_dim, output_dim, lr, activation='tanh'):
        np.random.seed(0)
        self.lr = lr  # Learning rate
        self.activation_fn = activation  # Activation function
        # Initialize weights with He initialization
        self.W1 = np.random.randn(input_dim, hidden_dim) * np.sqrt(2 / input_dim)
        self.b1 = np.zeros((1, hidden_dim))
        self.W2 = np.random.randn(hidden_dim, output_dim) * np.sqrt(2 / hidden_dim)
        self.b2 = np.zeros((1, output_dim))

    def forward(self, X):
        self.Z1 = np.dot(X, self.W1) + self.b1
        if self.activation_fn == 'tanh':
            self.A1 = np.tanh(self.Z1)
        elif self.activation_fn == 'relu':
            self.A1 = np.maximum(0, self.Z1)
        elif self.activation_fn == 'sigmoid':
            self.A1 = 1 / (1 + np.exp(-self.Z1))
            self.A1 = np.clip(self.A1, 1e-7, 1 - 1e-7)  # Prevent extreme values
        else:
            raise ValueError("Invalid activation function")
        self.Z2 = np.dot(self.A1, self.W2) + self.b2
        out = 1 / (1 + np.exp(-self.Z2))
        out = np.clip(out, 1e-7, 1 - 1e-7)  # Prevent extreme values
        return out

    def backward(self, X, y):
        m = X.shape[0]
        out = 1 / (1 + np.exp(-self.Z2))
        out = np.clip(out, 1e-7, 1 - 1e-7)  # Prevent extreme values
        dZ2 = out - y
        dW2 = np.dot(self.A1.T, dZ2) / m
        db2 = np.sum(dZ2, axis=0, keepdims=True) / m
        if self.activation_fn == 'tanh':
            dA1 = (1 - np.tanh(self.Z1) ** 2) * np.dot(dZ2, self.W2.T)
        elif self.activation_fn == 'relu':
            dA1 = (self.Z1 > 0).astype(float) * np.dot(dZ2, self.W2.T)
        elif self.activation_fn == 'sigmoid':
            sigmoid = 1 / (1 + np.exp(-self.Z1))
            sigmoid = np.clip(sigmoid, 1e-7, 1 - 1e-7)
            dA1 = sigmoid * (1 - sigmoid) * np.dot(dZ2, self.W2.T)
        dW1 = np.dot(X.T, dA1) / m
        db1 = np.sum(dA1, axis=0, keepdims=True) / m
        self.W2 -= self.lr * dW2
        self.b2 -= self.lr * db2
        self.W1 -= self.lr * dW1
        self.b1 -= self.lr * db1

    

def generate_data(n_samples=100):
    np.random.seed(0)
    X = np.random.randn(n_samples, 2)
    y = (X[:, 0] ** 2 + X[:, 1] ** 2 > 1).astype(int).reshape(-1, 1)
    return X, y

def plot_network(ax, mlp):
    # Draw neural network structure
    nodes = {'x1': [0.2, 0.8], 'x2': [0.2, 0.6], 'h1': [0.6, 0.8], 'h2': [0.6, 0.6], 'h3': [0.6, 0.4], 'y': [0.9, 0.7]}
    edges = [('x1', 'h1'), ('x1', 'h2'), ('x1', 'h3'), ('x2', 'h1'), ('x2', 'h2'), ('x2', 'h3'), ('h1', 'y'), ('h2', 'y'), ('h3', 'y')]
    for n1, n2 in edges:
        ax.plot([nodes[n1][0], nodes[n2][0]], [nodes[n1][1], nodes[n2][1]], color='purple', alpha=0.5, linewidth=2)
    for node in nodes:
        ax.scatter(nodes[node][0], nodes[node][1], color='blue', s=300)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')

def update(frame, mlp, ax_input, ax_hidden, ax_gradient, X, y):
    ax_hidden.clear()
    ax_input.clear()
    ax_gradient.clear()

    # Perform a few training steps
    for _ in range(10):
        mlp.forward(X)
        mlp.backward(X, y)

    # Plot hidden space features
    hidden_features = mlp.A1  # Hidden layer activations
    ax_hidden.scatter(
        hidden_features[:, 0],
        hidden_features[:, 1],
        c=y.ravel(),
        cmap='bwr',
        alpha=0.7
    )
    ax_hidden.set_title(f"Hidden Space at Step {frame}")

    # Compute decision boundary in the hidden space
    x1_range = np.linspace(hidden_features[:, 0].min(), hidden_features[:, 0].max(), 50)
    x2_range = np.linspace(hidden_features[:, 1].min(), hidden_features[:, 1].max(), 50)
    xx1, xx2 = np.meshgrid(x1_range, x2_range)
    grid = np.c_[xx1.ravel(), xx2.ravel()]
    zz = -(mlp.W2[0, 0] * xx1 + mlp.W2[1, 0] * xx2 + mlp.b2[0, 0]) / mlp.W2[2, 0]
    zz = zz.reshape(xx1.shape)

    # Add the decision boundary plane to the hidden space graph
    ax_hidden.plot_surface(
        xx1,
        xx2,
        zz,
        alpha=0.3,
        cmap='coolwarm'
    )

    # Plot input space decision boundary
    x1_range = np.linspace(-3, 3, 100)
    x2_range = np.linspace(-3, 3, 100)
    xx1, xx2 = np.meshgrid(x1_range, x2_range)
    grid = np.c_[xx1.ravel(), xx2.ravel()]
    decision_input = mlp.forward(grid).reshape(xx1.shape)
    ax_input.contourf(xx1, xx2, decision_input, levels=np.linspace(0, 1, 100), cmap='bwr', alpha=0.7)
    ax_input.scatter(X[:, 0], X[:, 1], c=y.ravel(), cmap='bwr', edgecolor='k')
    ax_input.set_title(f"Input Space at Step {frame}")

    # Plot network structure
    plot_network(ax_gradient, mlp)
    ax_gradient.set_title(f"Gradients at Step {frame}")


def visualize(activation, lr, step_num):
    X, y = generate_data()
    mlp = MLP(input_dim=2, hidden_dim=3, output_dim=1, lr=lr, activation=activation)
    fig = plt.figure(figsize=(21, 7))
    ax_hidden = fig.add_subplot(131, projection='3d')
    ax_input = fig.add_subplot(132)
    ax_gradient = fig.add_subplot(133)
    ani = FuncAnimation(fig, partial(update, mlp=mlp, ax_input=ax_input, ax_hidden=ax_hidden, ax_gradient=ax_gradient, X=X, y=y), frames=step_num // 10, repeat=False)
    ani.save(os.path.join(result_dir, "visualize.gif"), writer='pillow', fps=10)
    plt.close()

if __name__ == "__main__":
    visualize(activation="tanh", lr=0.1, step_num=1000)
