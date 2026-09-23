import numpy as np
from Projeto_Solar_RNA.network_config import load_network_config


def spatial_receptive_field(weights, shape, mask_shape=None):
    pixel_count = shape[0] * shape[1]
    aggregate_end = pixel_count + sum(shape)
    mask_shape = mask_shape if mask_shape is not None else load_network_config().mask_shape
    mask_rows, mask_columns = mask_shape
    mask_count = (shape[0] // mask_rows) * (shape[1] // mask_columns)
    if len(weights) in (aggregate_end, aggregate_end + mask_count):
        row_weights = weights[pixel_count:pixel_count + shape[0]]
        column_weights = weights[pixel_count + shape[0]:aggregate_end]
        field = (weights[:pixel_count].reshape(shape)
                 + row_weights[:, None] / shape[1]
                 + column_weights[None, :] / shape[0])
        if len(weights) > aggregate_end:
            mask_weights = weights[aggregate_end:].reshape(shape[0] // mask_rows, shape[1] // mask_columns)
            field += np.repeat(np.repeat(mask_weights, mask_rows, axis=0), mask_columns, axis=1) / (mask_rows * mask_columns)
        return field
    if len(weights) == pixel_count:
        return weights.reshape(shape)
    return weights


def softmax(x):
    """Numerically stable softmax."""
    x_shift = x - np.max(x)
    exp_x = np.exp(x_shift)
    return exp_x / np.sum(exp_x)


def relu(x):
    return np.maximum(0, x)


def relu_derivative(x):
    return (x > 0).astype(np.float32)


def sigmoid(x):
    x_clip = np.clip(x, -50.0, 50.0)
    return 1.0 / (1.0 + np.exp(-x_clip))


def sigmoid_derivative(x):
    s = sigmoid(x)
    return s * (1.0 - s)


class NeuralNetworkBase:
    """Base class for visual inspection & spatial audit of neural networks."""
    def __init__(self, num_inputs=None, num_classes=None, seed=None, config=None):
        self.config = config if config is not None else load_network_config()
        self.num_inputs = self.config.num_inputs if num_inputs is None else num_inputs
        self.num_classes = self.config.num_classes if num_classes is None else num_classes
        self.rng = np.random.default_rng(self.config.seed if seed is None else seed)
        
        self.last_input = None
        self.last_target = None
        self.last_scores = None
        self.last_probs = None
        self.last_prediction = None
        self.last_loss = 0.0
        self.last_update_had_error = False

    def reset_weights(self, seed=None):
        raise NotImplementedError

    def forward(self, x):
        raise NotImplementedError

    def train_step(self, x, target, lr=0.01):
        raise NotImplementedError


class SingleLayerPerceptron(NeuralNetworkBase):
    """
    Multiclass Perceptron with configured input features and output classes.
    No hidden layers. Uses pixels, row/column means and block means.
    """
    def __init__(self, num_inputs=None, num_classes=None, seed=None, config=None):
        super().__init__(num_inputs, num_classes, seed, config)
        self.weights = np.zeros((self.num_classes, self.num_inputs), dtype=np.float32)
        self.biases = np.zeros(self.num_classes, dtype=np.float32)

    def reset_weights(self, seed=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.weights.fill(0.0)
        self.biases.fill(0.0)

    def forward(self, x):
        self.last_input = np.asarray(x, dtype=np.float32)
        self.last_scores = self.weights @ self.last_input + self.biases
        self.last_probs = softmax(self.last_scores)
        self.last_prediction = int(np.argmax(self.last_scores))
        return self.last_prediction, self.last_scores, self.last_probs

    def train_step_phase1_forward(self, x, target):
        self.last_target = int(target)
        self.forward(x)
        return self.last_prediction, self.last_scores

    def train_step_phase2_eval(self):
        self.last_loss = -np.log(max(self.last_probs[self.last_target], 1e-12))
        self.last_update_had_error = (self.last_prediction != self.last_target)
        return self.last_update_had_error, self.last_loss

    def train_step_phase3_update(self, lr=0.01):
        if self.last_update_had_error:
            y_correct = self.last_target
            y_pred = self.last_prediction
            x = self.last_input
            
            self.weights[y_correct] += lr * x
            self.biases[y_correct] += lr
            
            self.weights[y_pred] -= lr * x
            self.biases[y_pred] -= lr

    def train_step(self, x, target, lr=0.01):
        self.train_step_phase1_forward(x, target)
        self.train_step_phase2_eval()
        self.train_step_phase3_update(lr)
        return self.last_prediction, self.last_update_had_error, self.last_loss

    def get_pixel_weights(self, r, c, img_shape=(40, 24)):
        """Returns outgoing weights from pixel (r, c) to all 12 classes."""
        idx = r * img_shape[1] + c
        return self.weights[:, idx]

    def get_receptive_field(self, output_idx, shape=(40, 24)):
        """Returns 2D heatmap matrix of weights for an output neuron."""
        w = self.weights[output_idx]
        return spatial_receptive_field(w, shape, self.config.mask_shape)


class MLPSolarNetwork(NeuralNetworkBase):
    """
    Multi-Layer Perceptron built from the shared network configuration.
    Trained via Backpropagation with Cross-Entropy Loss.
    """
    def __init__(self, num_inputs=None, hidden_grid=None, num_classes=None, activation=None, seed=None, config=None):
        super().__init__(num_inputs, num_classes, seed, config)
        hidden_grid = self.config.hidden_grid if hidden_grid is None else hidden_grid
        
        if isinstance(hidden_grid, tuple):
            self.hidden_rows, self.hidden_cols = hidden_grid
            self.num_hidden = self.hidden_rows * self.hidden_cols
        else:
            self.num_hidden = int(hidden_grid)
            self.hidden_rows, self.hidden_cols = 4, int(np.ceil(self.num_hidden / 4))
            self.num_hidden = self.hidden_rows * self.hidden_cols

        self.activation_name = self.config.activation if activation is None else activation
        
        # Layer 1 (Hidden Feature Detectors)
        self.W1 = None
        self.b1 = None
        
        # Layer 2 (Output Classes)
        self.W2 = None
        self.b2 = None
        
        # Internal activations
        self.z1 = None
        self.a1 = None
        self.z2 = None
        self.a2 = None
        
        self.reset_weights(seed)

    def reset_weights(self, seed=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
            
        scale1 = np.sqrt(2.0 / self.num_inputs)
        scale2 = np.sqrt(2.0 / self.num_hidden)
        
        self.W1 = self.rng.normal(0.0, scale1, size=(self.num_hidden, self.num_inputs)).astype(np.float32)
        self.b1 = np.zeros(self.num_hidden, dtype=np.float32)
        
        self.W2 = self.rng.normal(0.0, scale2, size=(self.num_classes, self.num_hidden)).astype(np.float32)
        self.b2 = np.zeros(self.num_classes, dtype=np.float32)

    def _activate(self, x):
        if self.activation_name == 'sigmoid':
            return sigmoid(x)
        return relu(x)

    def _activate_derivative(self, x):
        if self.activation_name == 'sigmoid':
            return sigmoid_derivative(x)
        return relu_derivative(x)

    def forward(self, x):
        self.last_input = np.asarray(x, dtype=np.float32)
        
        # Hidden layer forward
        self.z1 = self.W1 @ self.last_input + self.b1
        self.a1 = self._activate(self.z1)
        
        # Output layer forward
        self.z2 = self.W2 @ self.a1 + self.b2
        self.a2 = softmax(self.z2)
        
        self.last_scores = self.z2
        self.last_probs = self.a2
        self.last_prediction = int(np.argmax(self.a2))
        return self.last_prediction, self.last_scores, self.last_probs

    def train_step_phase1_forward(self, x, target):
        self.last_target = int(target)
        self.forward(x)
        return self.last_prediction, self.last_scores

    def train_step_phase2_eval(self):
        prob_target = max(self.last_probs[self.last_target], 1e-12)
        self.last_loss = -np.log(prob_target)
        self.last_update_had_error = (self.last_prediction != self.last_target)
        return self.last_update_had_error, self.last_loss

    def train_step_phase3_update(self, lr=0.01):
        y_onehot = np.zeros(self.num_classes, dtype=np.float32)
        y_onehot[self.last_target] = 1.0
        
        delta2 = self.a2 - y_onehot
        grad_W2 = np.outer(delta2, self.a1)
        grad_b2 = delta2
        
        delta1 = (self.W2.T @ delta2) * self._activate_derivative(self.z1)
        grad_W1 = np.outer(delta1, self.last_input)
        grad_b1 = delta1
        
        self.W2 -= lr * grad_W2
        self.b2 -= lr * grad_b2
        self.W1 -= lr * grad_W1
        self.b1 -= lr * grad_b1

    def train_step(self, x, target, lr=0.01):
        self.train_step_phase1_forward(x, target)
        self.train_step_phase2_eval()
        self.train_step_phase3_update(lr)
        return self.last_prediction, self.last_update_had_error, self.last_loss

    def get_hidden_activations(self):
        return self.a1 if self.a1 is not None else np.zeros(self.num_hidden)

    def get_hidden_activation_grid(self):
        acts = self.get_hidden_activations()
        return acts.reshape(self.hidden_rows, self.hidden_cols)

    def get_pixel_weights(self, r, c, img_shape=(40, 24)):
        """Returns outgoing weights from pixel (r, c) to all hidden neurons."""
        idx = r * img_shape[1] + c
        return self.W1[:, idx]  # shape (num_hidden,)

    def get_hidden_neuron_receptive_field(self, hidden_idx, shape=(40, 24)):
        """Returns 2D heatmap of incoming weights for a specific hidden detector."""
        w = self.W1[hidden_idx]
        return spatial_receptive_field(w, shape, self.config.mask_shape)

    def get_receptive_field(self, output_idx, shape=(40, 24)):
        """Computes effective input receptive field for output neuron (W2_i @ W1)."""
        w_eff = self.W2[output_idx] @ self.W1
        return spatial_receptive_field(w_eff, shape, self.config.mask_shape)


def evaluate_network(model, dataset, indices, mode='grayscale'):
    correct = 0
    total_loss = 0.0
    predictions = []
    targets = []
    
    for idx in indices:
        x = dataset.get_feature_vector(idx, mode=mode)
        target = dataset.labels[idx]
        pred, scores, probs = model.forward(x)
        
        loss = -np.log(max(probs[target], 1e-12))
        total_loss += loss
        if pred == target:
            correct += 1
            
        predictions.append(pred)
        targets.append(target)
        
    acc = correct / max(len(indices), 1)
    avg_loss = total_loss / max(len(indices), 1)
    return acc, avg_loss, np.array(predictions), np.array(targets)
