import random
import math

# =====================================================================
# 1. Andrej Karpathy's Micrograd Autograd Engine & MLP
# =====================================================================

class Value:
    """ Stores a scalar value and its gradient for backpropagation """
    def __init__(self, data, _children=(), _op=''):
        self.data = data
        self.grad = 0.0
        self._backward = lambda: None
        self._prev = set(_children)

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), '+')
        def _backward():
            self.grad += out.grad
            other.grad += out.grad
        out._backward = _backward
        return out

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), '*')
        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out._backward = _backward
        return out

    def __pow__(self, other):
        assert isinstance(other, (int, float)), "only supporting int/float powers"
        out = Value(self.data**other, (self,), f'**{other}')
        def _backward():
            self.grad += (other * (self.data ** (other - 1))) * out.grad
        out._backward = _backward
        return out

    def __sub__(self, other):
        return self + (other * -1)

    def __radd__(self, other):
        return self + other

    def tanh(self):
        x = self.data
        t = (math.exp(2*x) - 1) / (math.exp(2*x) + 1)
        out = Value(t, (self,), 'tanh')
        def _backward():
            self.grad += (1 - t**2) * out.grad
        out._backward = _backward
        return out

    def backward(self):
        topo = []
        visited = set()
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)
        build_topo(self)
        self.grad = 1.0
        for v in reversed(topo):
            v._backward()

    def __repr__(self):
        return f"Value(data={self.data:.4f}, grad={self.grad:.4f})"


class Neuron:
    def __init__(self, nin):
        self.w = [Value(random.uniform(-1, 1)) for _ in range(nin)]
        self.b = Value(random.uniform(-1, 1))

    def __call__(self, x):
        act = sum((wi * xi for wi, xi in zip(self.w, x)), self.b)
        return act.tanh()

    def parameters(self):
        return self.w + [self.b]


class Layer:
    def __init__(self, nin, nout):
        self.neurons = [Neuron(nin) for _ in range(nout)]

    def __call__(self, x):
        outs = [n(x) for n in self.neurons]
        return outs[0] if len(outs) == 1 else outs

    def parameters(self):
        return [p for neuron in self.neurons for p in neuron.parameters()]


class StandardMLP:
    """ Standard Multi-Layer Perceptron (Layer 1: 2 Neurons, Layer 2: 1 Neuron) """
    def __init__(self, nin, nouts):
        sz = [nin] + nouts
        self.layers = [Layer(sz[i], sz[i+1]) for i in range(len(nouts))]

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]


class CascadedMLP:
    """ Cascaded Architecture (Layer 1: 1 Neuron, Layer 2: 1 Neuron with Skip-Connections) """
    def __init__(self):
        self.layer1 = Layer(2, 1)  # Takes (x1, x2) -> outputs h1
        self.layer2 = Layer(3, 1)  # Takes (x1, x2, h1) -> outputs final XOR

    def __call__(self, x):
        h1 = self.layer1(x)
        # Skip connection: pass raw inputs [x1, x2] alongside h1 into Layer 2
        x2_input = [x[0], x[1], h1]
        return self.layer2(x2_input)

    def parameters(self):
        return self.layer1.parameters() + self.layer2.parameters()


# =====================================================================
# 2. XOR Dataset (-1 for 0, +1 for 1 to suit tanh activation range)
# =====================================================================

xs = [
    [-1.0, -1.0],  # (0, 0) -> Target: 0 (-1)
    [-1.0,  1.0],  # (0, 1) -> Target: 1 (+1)
    [ 1.0, -1.0],  # (1, 0) -> Target: 1 (+1)
    [ 1.0,  1.0]   # (1, 1) -> Target: 0 (-1)
]
ys = [-1.0, 1.0, 1.0, -1.0]


# =====================================================================
# 3. Training Function
# =====================================================================

def train_model(model, name, epochs=2000, lr=0.2):
    print(f"\n==========================================")
    print(f" Training: {name}")
    print(f" Total Parameters: {len(model.parameters())}")
    print(f"==========================================")

    for k in range(epochs):
        # 1. Forward pass
        ypred = [model(x) for x in xs]
        
        # 2. Compute Loss (Mean Squared Error)
        loss = sum((yout - ygt)**2 for ygt, yout in zip(ys, ypred))

        # 3. Zero out gradients
        for p in model.parameters():
            p.grad = 0.0

        # 4. Backward pass (Autograd)
        loss.backward()

        # 5. Stochastic Gradient Descent (SGD) Update
        for p in model.parameters():
            p.data -= lr * p.grad

        if k % 40 == 0 or k == epochs - 1:
            print(f"Epoch {k:3d} | Loss: {loss.data:.4f}")

    print("\n--- Final Predictions ---")
    for x, target in zip(xs, ys):
        pred = model(x).data
        # Convert inputs/targets back to binary (0/1) for clean reading
        bin_x = [1 if v > 0 else 0 for v in x]
        bin_target = 1 if target > 0 else 0
        bin_pred = 1 if pred > 0 else 0
        print(f"XOR({bin_x[0]}, {bin_x[1]}) -> Target: {bin_target} | Raw Output: {pred:+.3f} (Predicted Class: {bin_pred})")


# =====================================================================
# 4. Run Both XOR Implementations
# =====================================================================

# Method 1: Standard (Layer 1: 2 Neurons, Layer 2: 1 Neuron)
standard_model = StandardMLP(2, [2, 1])
train_model(standard_model, "Method 1: Standard (2 Neurons in Layer 1, 1 in Layer 2)")

# Method 2: Cascaded (Layer 1: 1 Neuron, Layer 2: 1 Neuron with Skip Connections)
cascaded_model = CascadedMLP()
train_model(cascaded_model, "Method 2: Cascaded Deep (1 Neuron in Layer 1, 1 in Layer 2)")