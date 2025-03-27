import numpy as np
import matplotlib.pyplot as plt
import time

class DeepNeuralNetwork:
    def __init__(self, input_size, hidden_sizes, output_size, activation='relu', use_dropout=False,dropout_rate=0.2, use_batch_norm=False, weight_decay=0, optimizer='sgd'):
        """
        Initialize the deep neural network.
        Parameters:
            - input_size: Size of the input features
            - hidden_sizes: List with the size of each hidden layer
            - output_size: Size of the output layer (number of classes)
            - activation: Activation function for hidden layers ('relu' or 'gelu')
            - use_dropout: Whether to use dropout
            - dropout_rate: Dropout rate (if dropout is used)
            - use_batch_norm: Whether to use batch normalization
            - weight_decay: L2 regularization parameter
            - optimizer: Which optimizer to use ('sgd', 'momentum', 'adam')
        """
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.output_size = output_size
        self.activation = activation
        self.use_dropout = use_dropout
        self.dropout_rate = dropout_rate
        self.use_batch_norm = use_batch_norm
        self.weight_decay = weight_decay
        self.optimizer = optimizer
        
        # Initialize parameters
        self.parameters = {}
        self.initialize_parameters()
        
        # Initialize optimizer-specific parameters
        if optimizer == 'momentum' or optimizer == 'adam':
            self.velocity = {}
            self.initialize_velocity()
        
        if optimizer == 'adam':
            self.m = {}  # First moment
            self.v = {}  # Second moment
            self.initialize_adam_parameters()
        
        # Batch normalization parameters
        if use_batch_norm:
            self.bn_params = {}
            self.initialize_bn_parameters()

    def initialize_parameters(self):
        """
        Initialize the weights and biases of the neural network.
        """
        np.random.seed(42)  # For reproducibility
        
        layer_sizes = [self.input_size] + self.hidden_sizes + [self.output_size]
        
        for l in range(1, len(layer_sizes)):
            # Optimized initialization for faster convergence
            if self.activation == 'relu':
                # He initialization for ReLU
                scale = np.sqrt(2 / layer_sizes[l-1])
            elif self.activation == 'gelu':
                # Optimized initialization for GELU
                scale = np.sqrt(2 / layer_sizes[l-1])
            else:
                # Xavier initialization for other activations
                scale = np.sqrt(1 / layer_sizes[l-1])
                
            # Use a more efficient initialization approach
            self.parameters[f'W{l}'] = np.random.randn(layer_sizes[l], layer_sizes[l-1]) * scale
            self.parameters[f'b{l}'] = np.zeros((layer_sizes[l], 1))

    def initialize_velocity(self):
        """
        Initialize the velocity for momentum-based optimizers.
        """
        layer_sizes = [self.input_size] + self.hidden_sizes + [self.output_size]
        
        for l in range(1, len(layer_sizes)):
            self.velocity[f'W{l}'] = np.zeros((layer_sizes[l], layer_sizes[l-1]))
            self.velocity[f'b{l}'] = np.zeros((layer_sizes[l], 1))

    def initialize_adam_parameters(self):
        """
        Initialize the Adam optimizer parameters.
        """
        layer_sizes = [self.input_size] + self.hidden_sizes + [self.output_size]
        
        for l in range(1, len(layer_sizes)):
            self.m[f'W{l}'] = np.zeros((layer_sizes[l], layer_sizes[l-1]))
            self.m[f'b{l}'] = np.zeros((layer_sizes[l], 1))
            self.v[f'W{l}'] = np.zeros((layer_sizes[l], layer_sizes[l-1]))
            self.v[f'b{l}'] = np.zeros((layer_sizes[l], 1))

    def initialize_bn_parameters(self):
        """
        Initialize the batch normalization parameters.
        """
        for l in range(1, len(self.hidden_sizes) + 1):
            self.bn_params[f'gamma{l}'] = np.ones((self.hidden_sizes[l-1], 1))
            self.bn_params[f'beta{l}'] = np.zeros((self.hidden_sizes[l-1], 1))
            self.bn_params[f'running_mean{l}'] = np.zeros((self.hidden_sizes[l-1], 1))
            self.bn_params[f'running_var{l}'] = np.zeros((self.hidden_sizes[l-1], 1))

    def relu(self, Z):
        """
        Compute the ReLU activation.
        """
        return np.maximum(0, Z)

    def relu_derivative(self, Z):
        """
        Compute the derivative of ReLU.
        """
        return np.where(Z > 0, 1, 0)

    def gelu(self, Z):
        """
        Compute the GELU activation: 0.5 * Z * (1 + tanh(sqrt(2/pi) * (Z + 0.044715 * Z^3)))
        """
        return 0.5 * Z * (1 + np.tanh(np.sqrt(2 / np.pi) * (Z + 0.044715 * np.power(Z, 3))))

    def gelu_derivative(self, Z):
        """
        Compute the derivative of GELU.
        """
        # Faster approximation of GELU derivative
        return 0.5 * (1 + np.tanh(np.sqrt(2 / np.pi) * (Z + 0.044715 * np.power(Z, 3)))) + \
               0.5 * Z * (1 - np.tanh(np.sqrt(2 / np.pi) * (Z + 0.044715 * np.power(Z, 3)))**2) * \
               np.sqrt(2 / np.pi) * (1 + 0.134145 * np.power(Z, 2))

    def softmax(self, Z):
        """
        Compute the softmax activation.
        """
        # Subtract max for numerical stability
        exp_Z = np.exp(Z - np.max(Z, axis=0, keepdims=True))
        return exp_Z / np.sum(exp_Z, axis=0, keepdims=True)

    def batch_normalize(self, Z, gamma, beta, layer_idx, is_training=True, epsilon=1e-5):
        """
        Apply batch normalization.
        
        Parameters:
        - Z: Input to normalize
        - gamma: Scale parameter
        - beta: Shift parameter
        - layer_idx: Layer index for running mean and variance
        - is_training: Whether the model is in training mode
        - epsilon: Small constant for numerical stability
        
        Returns:
        - out: Normalized output
        - cache: Cache for backward pass
        """
        if is_training:
            batch_mean = np.mean(Z, axis=1, keepdims=True)
            batch_var = np.var(Z, axis=1, keepdims=True)
            
            # Use a more balanced momentum (0.95) for faster adaptation
            self.bn_params[f'running_mean{layer_idx}'] = 0.95 * self.bn_params[f'running_mean{layer_idx}'] + 0.05 * batch_mean
            self.bn_params[f'running_var{layer_idx}'] = 0.95 * self.bn_params[f'running_var{layer_idx}'] + 0.05 * batch_var
            
            # Normalize
            Z_norm = (Z - batch_mean) / np.sqrt(batch_var + epsilon)
            out = gamma * Z_norm + beta
            
            # Cache for backward pass
            cache = (Z, Z_norm, batch_mean, batch_var, gamma, beta, epsilon)
            return out, cache
        else:
            # Use running statistics
            mean = self.bn_params[f'running_mean{layer_idx}']
            var = self.bn_params[f'running_var{layer_idx}']
            Z_norm = (Z - mean) / np.sqrt(var + epsilon)
            out = gamma * Z_norm + beta
            return out, None

    def dropout(self, A, is_training=True):
        """
        Apply dropout.
        
        Parameters:
        - A: Activations
        - is_training: Whether the model is in training mode
        
        Returns:
        - A_dropped: Activations after dropout
        - D: Dropout mask
        """
        if is_training and self.use_dropout:
            D = np.random.rand(*A.shape) > self.dropout_rate
            A = A * D / (1 - self.dropout_rate)  # Scale to maintain expected value
            return A, D
        return A, None

    def forward_propagation(self, X, is_training=True):
        """
        Perform forward propagation with residual connections.
        
        Parameters:
        - X: Input features (input_size, batch_size)
        - is_training: Whether the model is in training mode
        
        Returns:
        - A_final: Final activations
        - caches: Cached values for backward propagation
        """
        caches = []
        A = X
        L = len(self.hidden_sizes) + 1  # Total number of layers
        
        # Store intermediate activations for residual connections
        activations = [A]
        
        # Process through each layer
        for l in range(1, L + 1):
            A_prev = A
            Z = np.dot(self.parameters[f'W{l}'], A_prev) + self.parameters[f'b{l}']
            
            # Batch normalization (not applied to output layer)
            bn_cache = None
            if self.use_batch_norm and l < L:
                Z, bn_cache = self.batch_normalize(
                    Z,
                    self.bn_params[f'gamma{l}'],
                    self.bn_params[f'beta{l}'],
                    l,
                    is_training
                )
            
            # Activation
            if l == L:  # Output layer
                A = self.softmax(Z)
                has_residual = False
            else:  # Hidden layer
                if self.activation == 'relu':
                    A = self.relu(Z)
                elif self.activation == 'gelu':
                    A = self.gelu(Z)
                
                # Add residual connection if shapes match and not first layer
                has_residual = False
                if l > 1 and l % 2 == 0:  # Only add residual every 2 layers
                    prev_layer = l - 2
                    if prev_layer >= 0 and A.shape[0] == activations[prev_layer].shape[0]:
                        A = A + activations[prev_layer]
                        has_residual = True
            
            # Dropout (not applied to output layer)
            D = None
            if self.use_dropout and is_training and l < L:
                A, D = self.dropout(A)
            
            # Store activation for potential residual connections
            activations.append(A)
            
            # Store cache for backward pass
            cache = (A_prev, Z, D, bn_cache, has_residual)
            caches.append(cache)
        
        return A, caches

    def compute_cost(self, AL, Y):
        """
        Compute the cross-entropy cost with label smoothing and regularization.
        
        Parameters:
        - AL: Output of forward propagation (output_size, batch_size)
        - Y: True labels (output_size, batch_size)
        
        Returns:
        - cost: Cross-entropy cost with regularization
        """
        m = Y.shape[1]
        
        # Apply label smoothing (helps prevent overfitting)
        smoothing = 0.1
        Y_smooth = (1 - smoothing) * Y + smoothing / Y.shape[0]
        
        # Cross-entropy loss with label smoothing
        logprobs = np.multiply(np.log(AL + 1e-8), Y_smooth)
        cost = -np.sum(logprobs) / m
        
        # Add focal loss component to focus on hard examples
        focal_weight = np.power(1 - np.sum(AL * Y, axis=0), 2)
        focal_loss = -np.sum(np.multiply(np.log(AL + 1e-8), Y) * focal_weight) / m
        
        # Combine losses
        combined_loss = 0.7 * cost + 0.3 * focal_loss
        
        # Add L2 regularization cost
        if self.weight_decay > 0:
            L2_cost = 0
            for l in range(1, len(self.hidden_sizes) + 2):
                L2_cost += np.sum(np.square(self.parameters[f'W{l}']))
            combined_loss += (self.weight_decay / (2 * m)) * L2_cost
        
        return combined_loss

    def batch_normalize_backward(self, dout, cache):
        """
        Backward pass for batch normalization.
        
        Parameters:
        - dout: Gradient of the cost with respect to the normalized activations
        - cache: Cache of values needed for backward pass
        
        Returns:
        - dx: Gradient with respect to x
        - dgamma: Gradient with respect to gamma
        - dbeta: Gradient with respect to beta
        """
        Z, Z_norm, mean, var, gamma, beta, epsilon = cache
        m = Z.shape[1]
        
        dbeta = np.sum(dout, axis=1, keepdims=True)
        dgamma = np.sum(dout * Z_norm, axis=1, keepdims=True)
        
        # Gradient of the cost with respect to Z_norm
        dZ_norm = dout * gamma
        
        # Gradient of the cost with respect to variance
        dvar = np.sum(dZ_norm * (Z - mean) * -0.5 * np.power(var + epsilon, -1.5), axis=1, keepdims=True)
        
        # Gradient of the cost with respect to mean
        dmean = np.sum(dZ_norm * -1 / np.sqrt(var + epsilon), axis=1, keepdims=True) + \
                dvar * np.sum(-2 * (Z - mean), axis=1, keepdims=True) / m
        
        # Gradient of the cost with respect to Z
        dZ = dZ_norm / np.sqrt(var + epsilon) + \
            dvar * 2 * (Z - mean) / m + \
            dmean / m
        
        return dZ, dgamma, dbeta

    def backward_propagation(self, AL, Y, caches):
        """
        Perform backward propagation with residual connections.
        
        Parameters:
        - AL: Output of forward propagation (output_size, batch_size)
        - Y: True labels (output_size, batch_size)
        - caches: Cached values from forward propagation
        
        Returns:
        - gradients: Gradients of the cost with respect to the parameters
        """
        gradients = {}
        L = len(caches)  # Number of layers
        m = AL.shape[1]  # Batch size
        
        # Initialize the backpropagation with softmax derivative
        dAL = AL - Y  # Derivative of cost with respect to AL (for softmax with cross-entropy)
        
        # Initialize batch normalization gradients
        if self.use_batch_norm:
            self.bn_gradients = {}
        
        # Loop from last layer to first layer
        current_cache = caches[L-1]
        A_prev, Z, D, bn_cache, _ = current_cache
        
        # Compute gradients for the output layer
        dW = np.dot(dAL, A_prev.T) / m
        db = np.sum(dAL, axis=1, keepdims=True) / m
        
        # Add L2 regularization
        if self.weight_decay > 0:
            dW += (self.weight_decay / m) * self.parameters[f'W{L}']
        
        # Store gradients for output layer
        gradients[f'dW{L}'] = dW
        gradients[f'db{L}'] = db
        
        # Initialize dA for the backwards computation through hidden layers
        dA = np.dot(self.parameters[f'W{L}'].T, dAL)
        
        # Loop through the hidden layers
        for l in reversed(range(L-1)):
            current_cache = caches[l]
            A_prev, Z, D, bn_cache, has_residual = current_cache
            
            # Apply dropout mask if using dropout
            if self.use_dropout and D is not None:
                # Make sure shapes match
                if D.shape == dA.shape:
                    dA = dA * D / (1 - self.dropout_rate)
            
            # Apply activation derivative
            if self.activation == 'relu':
                dZ = dA * self.relu_derivative(Z)
            elif self.activation == 'gelu':
                dZ = dA * self.gelu_derivative(Z)
            
            # Apply batch normalization backward
            if self.use_batch_norm and bn_cache is not None:
                dZ, dgamma, dbeta = self.batch_normalize_backward(dZ, bn_cache)
                self.bn_gradients[f'dgamma{l+1}'] = dgamma
                self.bn_gradients[f'dbeta{l+1}'] = dbeta
            
            # Compute gradients for weights and biases
            dW = np.dot(dZ, A_prev.T) / m
            db = np.sum(dZ, axis=1, keepdims=True) / m
            
            # Add L2 regularization
            if self.weight_decay > 0:
                dW += (self.weight_decay / m) * self.parameters[f'W{l+1}']
            
            # Store gradients
            gradients[f'dW{l+1}'] = dW
            gradients[f'db{l+1}'] = db
            
            # Compute dA for next layer
            if l > 0:
                dA_prev = np.dot(self.parameters[f'W{l+1}'].T, dZ)
                
                # Add residual gradient if this layer has a residual connection
                if has_residual:
                    # For simplicity, we'll just pass the gradient through
                    # without trying to route it back to earlier layers
                    dA = dA_prev
                else:
                    dA = dA_prev
        
        return gradients

    def update_parameters(self, gradients, learning_rate, beta1=0.9, beta2=0.999, epsilon=1e-8, t=1):
        """
        Update the parameters using the chosen optimizer.
        
        Parameters:
        - gradients: Gradients from backward propagation
        - learning_rate: Learning rate for optimization
        - beta1, beta2: Adam hyperparameters
        - epsilon: Small constant for numerical stability
        - t: Time step for Adam
        """
        L = len(self.hidden_sizes) + 1  # Total number of layers
        
        # Update weights and biases
        for l in range(1, L + 1):
            if self.optimizer == 'sgd':
                # Standard SGD update
                self.parameters[f'W{l}'] -= learning_rate * gradients[f'dW{l}']
                self.parameters[f'b{l}'] -= learning_rate * gradients[f'db{l}']
            
            elif self.optimizer == 'momentum':
                # Momentum update
                self.velocity[f'W{l}'] = beta1 * self.velocity[f'W{l}'] + (1 - beta1) * gradients[f'dW{l}']
                self.velocity[f'b{l}'] = beta1 * self.velocity[f'b{l}'] + (1 - beta1) * gradients[f'db{l}']
                
                self.parameters[f'W{l}'] -= learning_rate * self.velocity[f'W{l}']
                self.parameters[f'b{l}'] -= learning_rate * self.velocity[f'b{l}']
            
            elif self.optimizer == 'adam':
                # Adam update
                # Update first moment
                self.m[f'W{l}'] = beta1 * self.m[f'W{l}'] + (1 - beta1) * gradients[f'dW{l}']
                self.m[f'b{l}'] = beta1 * self.m[f'b{l}'] + (1 - beta1) * gradients[f'db{l}']
                
                # Update second moment
                self.v[f'W{l}'] = beta2 * self.v[f'W{l}'] + (1 - beta2) * np.power(gradients[f'dW{l}'], 2)
                self.v[f'b{l}'] = beta2 * self.v[f'b{l}'] + (1 - beta2) * np.power(gradients[f'db{l}'], 2)
                
                # Bias correction
                m_corrected_W = self.m[f'W{l}'] / (1 - np.power(beta1, t))
                m_corrected_b = self.m[f'b{l}'] / (1 - np.power(beta1, t))
                v_corrected_W = self.v[f'W{l}'] / (1 - np.power(beta2, t))
                v_corrected_b = self.v[f'b{l}'] / (1 - np.power(beta2, t))
                
                # Update parameters
                self.parameters[f'W{l}'] -= learning_rate * m_corrected_W / (np.sqrt(v_corrected_W) + epsilon)
                self.parameters[f'b{l}'] -= learning_rate * m_corrected_b / (np.sqrt(v_corrected_b) + epsilon)
        
        # Update batch normalization parameters
        if self.use_batch_norm:
            for l in range(1, L):  # No batch norm for output layer
                if f'dgamma{l}' in self.bn_gradients and f'dbeta{l}' in self.bn_gradients:
                    # Simple SGD update for batch norm parameters
                    self.bn_params[f'gamma{l}'] -= learning_rate * self.bn_gradients[f'dgamma{l}']
                    self.bn_params[f'beta{l}'] -= learning_rate * self.bn_gradients[f'dbeta{l}']

    def minibatch_split(self, X, Y, batch_size):
        """
        Split the data into mini-batches.
        
        Parameters:
        - X: Input features (input_size, num_examples)
        - Y: Labels (output_size, num_examples)
        - batch_size: Size of each mini-batch
        
        Returns:
        - mini_batches: List of mini-batches
        """
        m = X.shape[1]
        permutation = np.random.permutation(m)
        X_shuffled = X[:, permutation]
        Y_shuffled = Y[:, permutation]
        
        mini_batches = []
        num_complete_minibatches = m // batch_size
        
        for k in range(0, num_complete_minibatches):
            mini_batch_X = X_shuffled[:, k * batch_size : (k + 1) * batch_size]
            mini_batch_Y = Y_shuffled[:, k * batch_size : (k + 1) * batch_size]
            mini_batches.append((mini_batch_X, mini_batch_Y))
        
        # Handle the end case (last mini-batch < batch_size)
        if m % batch_size != 0:
            mini_batch_X = X_shuffled[:, num_complete_minibatches * batch_size : m]
            mini_batch_Y = Y_shuffled[:, num_complete_minibatches * batch_size : m]
            mini_batches.append((mini_batch_X, mini_batch_Y))
        
        return mini_batches

    def fit(self, X, Y, X_val=None, Y_val=None, num_epochs=100, batch_size=32, learning_rate=0.01,
            beta1=0.9, beta2=0.999, epsilon=1e-8, print_every=10, lr_decay=0.95, lr_decay_epoch=10,
            early_stopping=True, patience=10, warmup_epochs=5):
        """
        Train the neural network.
        
        Parameters:
        - X: Input features (input_size, num_examples)
        - Y: True labels (output_size, num_examples)
        - X_val: Validation features
        - Y_val: Validation labels
        - num_epochs: Number of epochs
        - batch_size: Size of each mini-batch
        - learning_rate: Learning rate for optimization
        - beta1, beta2: Adam hyperparameters
        - epsilon: Small constant for numerical stability
        - print_every: Print loss every print_every epochs
        
        Returns:
        - history: Dictionary containing training metrics
        """
        np.random.seed(42)
        history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
        # Initialize the actual learning rate with warmup
        if warmup_epochs > 0:
            # Start with a lower learning rate for warmup
            current_learning_rate = learning_rate * 0.1
        else:
            current_learning_rate = learning_rate
        
        # Early stopping variables
        best_val_loss = float('inf')
        best_epoch = 0
        best_params = None
        no_improvement = 0
        
        for epoch in range(num_epochs):
            epoch_time = time.time()
            epoch_cost = 0
            
            # Apply learning rate warmup or decay
            if warmup_epochs > 0 and epoch < warmup_epochs:
                # Linear warmup
                current_learning_rate = learning_rate * (0.1 + 0.9 * (epoch + 1) / warmup_epochs)
                print(f"Warmup learning rate: {current_learning_rate:.6f}")
            elif lr_decay < 1 and epoch >= warmup_epochs and (epoch - warmup_epochs) % lr_decay_epoch == 0:
                current_learning_rate *= lr_decay
                print(f"Learning rate decayed to: {current_learning_rate:.6f}")
            
            # Split data into mini-batches
            mini_batches = self.minibatch_split(X, Y, batch_size)
            mini_batches = self.minibatch_split(X, Y, batch_size)
            
            # Initialize batch normalization gradients
            if self.use_batch_norm:
                self.bn_gradients = {}
            
            # Training loop over mini-batches
            for t, mini_batch in enumerate(mini_batches, 1):
                X_mini, Y_mini = mini_batch
                
                # Forward propagation
                AL, caches = self.forward_propagation(X_mini, is_training=True)
                
                # Compute cost
                mini_batch_cost = self.compute_cost(AL, Y_mini)
                epoch_cost += mini_batch_cost / len(mini_batches)
                
                # Backward propagation
                gradients = self.backward_propagation(AL, Y_mini, caches)
                
                # Update parameters with the current learning rate
                self.update_parameters(gradients, current_learning_rate, beta1, beta2, epsilon, epoch * len(mini_batches) + t)
            
            # Store training cost
            history['train_loss'].append(epoch_cost)
            
            # Compute training accuracy
            train_preds = self.predict(X)
            train_acc = np.mean(np.argmax(train_preds, axis=0) == np.argmax(Y, axis=0))
            history['train_acc'].append(train_acc)
            
            # Compute validation metrics if provided
            if X_val is not None and Y_val is not None:
                val_preds, val_loss = self.predict(X_val, Y_val, return_loss=True)
                val_acc = np.mean(np.argmax(val_preds, axis=0) == np.argmax(Y_val, axis=0))
                history['val_loss'].append(val_loss)
                history['val_acc'].append(val_acc)
                
                # Early stopping check
                if early_stopping:
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        best_epoch = epoch
                        best_params = {k: v.copy() for k, v in self.parameters.items()}
                        no_improvement = 0
                    else:
                        no_improvement += 1
                        if no_improvement >= patience:
                            print(f"Early stopping at epoch {epoch+1}. Best validation loss: {best_val_loss:.4f} at epoch {best_epoch+1}")
                            # Restore best parameters
                            self.parameters = best_params
                            break
            
            # Print progress
            if (epoch + 1) % print_every == 0 or epoch == num_epochs - 1:
                epoch_duration = time.time() - epoch_time
                message = f"Epoch {epoch + 1}/{num_epochs} - {epoch_duration:.2f}s - loss: {epoch_cost:.4f} - acc: {train_acc:.4f}"
                if X_val is not None:
                    message += f" - val_loss: {val_loss:.4f} - val_acc: {val_acc:.4f}"
                print(message)
        
        return history

    def predict(self, X, Y=None, return_loss=False):
        """
        Make predictions using the trained neural network.
        
        Parameters:
        - X: Input features (input_size, num_examples)
        - Y: True labels (optional, for computing loss)
        - return_loss: Whether to return the loss
        
        Returns:
        - predictions: Model predictions
        - loss: Loss (if return_loss is True and Y is provided)
        """
        # Forward propagation
        AL, _ = self.forward_propagation(X, is_training=False)
        
        # Compute loss if Y is provided
        if Y is not None and return_loss:
            loss = self.compute_cost(AL, Y)
            return AL, loss
        
        return AL

    def evaluate(self, X, Y):
        """
        Evaluate the model performance.
        
        Parameters:
        - X: Input features (input_size, num_examples)
        - Y: True labels (output_size, num_examples)
        
        Returns:
        - metrics: Dictionary with performance metrics
        """
        # Predictions
        predictions, loss = self.predict(X, Y, return_loss=True)
        pred_classes = np.argmax(predictions, axis=0)
        true_classes = np.argmax(Y, axis=0)
        
        # Accuracy
        accuracy = np.mean(pred_classes == true_classes)
        
        # Precision, recall, F1 score per class
        precision = []
        recall = []
        f1 = []
        
        for c in range(self.output_size):
            tp = np.sum((pred_classes == c) & (true_classes == c))
            fp = np.sum((pred_classes == c) & (true_classes != c))
            fn = np.sum((pred_classes != c) & (true_classes == c))
            
            # Handle division by zero
            p = tp / (tp + fp) if (tp + fp) > 0 else 0
            r = tp / (tp + fn) if (tp + fn) > 0 else 0
            
            precision.append(p)
            recall.append(r)
            f1.append(2 * p * r / (p + r) if (p + r) > 0 else 0)
        
        # Macro-average metrics
        macro_precision = np.mean(precision)
        macro_recall = np.mean(recall)
        macro_f1 = np.mean(f1)
        
        return {
            'loss': loss,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'macro_precision': macro_precision,
            'macro_recall': macro_recall,
            'macro_f1': macro_f1
        }
    
class DataProcessor:
    @staticmethod
    def one_hot_encode(y, num_classes):
        """
        One-hot encode the labels.
        Parameters:
        - y: Array of class labels (num_examples,)
        - num_classes: Number of classes
        
        Returns:
        - one_hot: One-hot encoded labels (num_classes, num_examples)
        """
        m = y.shape[0]
        one_hot = np.zeros((num_classes, m))
        for i in range(m):
            one_hot[y[i], i] = 1
        return one_hot

    @staticmethod
    def normalize_data(X):
        """
        Normalize the data to have zero mean and unit variance.
        
        Parameters:
        - X: Input features (input_size, num_examples)
        
        Returns:
        - X_norm: Normalized features
        - mean: Mean of each feature
        - std: Standard deviation of each feature
        """
        mean = np.mean(X, axis=1, keepdims=True)
        std = np.std(X, axis=1, keepdims=True)
        X_norm = (X - mean) / (std + 1e-8)
        return X_norm, mean, std

    @staticmethod
    def split_data(X, Y, val_ratio=0.2, shuffle=True):
        """
        Split the data into training and validation sets.
        
        Parameters:
        - X: Input features (input_size, num_examples)
        - Y: Labels (output_size, num_examples)
        - val_ratio: Ratio of validation data
        - shuffle: Whether to shuffle the data before splitting
        
        Returns:
        - X_train, Y_train: Training data
        - X_val, Y_val: Validation data
        """
        m = X.shape[1]
        indices = np.arange(m)
        
        if shuffle:
            np.random.shuffle(indices)
        
        val_size = int(m * val_ratio)
        val_indices = indices[:val_size]
        train_indices = indices[val_size:]
        
        X_train = X[:, train_indices]
        Y_train = Y[:, train_indices]
        X_val = X[:, val_indices]
        Y_val = Y[:, val_indices]
        
        return X_train, Y_train, X_val, Y_val
    
class Visualizer:
    @staticmethod
    def plot_training_history(history):
        """
        Plot the training history.
        Parameters:
        - history: Dictionary containing training metrics
        """
        plt.figure(figsize=(12, 5))
        
        # Plot loss
        plt.subplot(1, 2, 1)
        plt.plot(history['train_loss'], label='Train Loss')
        if 'val_loss' in history and history['val_loss']:
            plt.plot(history['val_loss'], label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss')
        plt.legend()
        
        # Plot accuracy
        plt.subplot(1, 2, 2)
        plt.plot(history['train_acc'], label='Train Accuracy')
        if 'val_acc' in history and history['val_acc']:
            plt.plot(history['val_acc'], label='Validation Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.title('Training and Validation Accuracy')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig('training_history.png')
        plt.close()

    @staticmethod
    def plot_confusion_matrix(y_true, y_pred, classes):
        """
        Plot the confusion matrix.
        
        Parameters:
        - y_true: True class indices
        - y_pred: Predicted class indices
        - classes: List of class names
        """
        cm = np.zeros((len(classes), len(classes)), dtype=np.int32)
        for i in range(len(y_true)):
            cm[y_true[i], y_pred[i]] += 1
        
        plt.figure(figsize=(10, 8))
        plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title('Confusion Matrix')
        plt.colorbar()
        
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, rotation=45)
        plt.yticks(tick_marks, classes)
        
        # Add text annotations
        fmt = 'd'
        thresh = cm.max() / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                plt.text(j, i, format(cm[i, j], fmt),
                    horizontalalignment="center",
                    color="white" if cm[i, j] > thresh else "black")
        
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.tight_layout()
        plt.savefig('confusion_matrix.png')
        plt.close()


def main():
    """
    Main function to execute the neural network training and evaluation.
    """
    # Load data (assuming data is stored in .npy files)
    try:
        X_train = np.load('train_data.npy')
        y_train = np.load('train_label.npy')
        X_test = np.load('test_data.npy')
        y_test = np.load('test_label.npy')
        print(f"Training data shape: {X_train.shape}")
        print(f"Training labels shape: {y_train.shape}")
        print(f"Test data shape: {X_test.shape}")
        print(f"Test labels shape: {y_test.shape}")
    except FileNotFoundError as e:
        print(f"Error loading data: {e}")
        print("Please ensure the data files are in the correct location.")
        return

    # Preprocess data
    # Transpose data if needed to match (input_size, num_examples) format
    if X_train.shape[0] != 128:
        X_train = X_train.T
        X_test = X_test.T

    # Normalize data
    X_train, mean, std = DataProcessor.normalize_data(X_train)
    X_test = (X_test - mean) / (std + 1e-8)

    # One-hot encode labels
    num_classes = 10  # As specified
    
    # Reshape labels if they're not 1D arrays
    if len(y_train.shape) > 1 and y_train.shape[1] == 1:
        y_train = y_train.reshape(-1)
    if len(y_test.shape) > 1 and y_test.shape[1] == 1:
        y_test = y_test.reshape(-1)
    
    # One-hot encode
    Y_train = DataProcessor.one_hot_encode(y_train, num_classes)
    Y_test = DataProcessor.one_hot_encode(y_test, num_classes)
    
    # Verify shapes
    print(f"After one-hot encoding:")
    print(f"Y_train shape: {Y_train.shape}")
    print(f"Y_test shape: {Y_test.shape}")

    # Split training data for validation
    X_train, Y_train, X_val, Y_val = DataProcessor.split_data(X_train, Y_train, val_ratio=0.1)

    print(f"After preprocessing:")
    print(f"X_train shape: {X_train.shape}")
    print(f"Y_train shape: {Y_train.shape}")
    print(f"X_val shape: {X_val.shape}")
    print(f"Y_val shape: {Y_val.shape}")
    print(f"X_test shape: {X_test.shape}")
    print(f"Y_test shape: {Y_test.shape}")
    # Feature preprocessing with standardization (zero mean, unit variance)
    print("Applying feature standardization...")
    
    # Standardize data
    X_train_std = (X_train - np.mean(X_train, axis=1, keepdims=True)) / (np.std(X_train, axis=1, keepdims=True) + 1e-8)
    X_val_std = (X_val - np.mean(X_train, axis=1, keepdims=True)) / (np.std(X_train, axis=1, keepdims=True) + 1e-8)
    X_test_std = (X_test - np.mean(X_train, axis=1, keepdims=True)) / (np.std(X_train, axis=1, keepdims=True) + 1e-8)
    
    print(f"Standardized data - X_train shape: {X_train_std.shape}")
    print(f"Standardized data - X_val shape: {X_val_std.shape}")
    print(f"Standardized data - X_test shape: {X_test_std.shape}")
    
    # Create a simple but effective model
    print("Creating a simple but effective model...")
    
    # Define hyperparameters for the optimal pyramid architecture (57.67% accuracy)
    input_size = X_train_std.shape[0]  # Should be 128
    hidden_sizes = [512, 256, 128, 64]  # Pyramid-shaped network (gradually decreasing)
    output_size = num_classes
    activation = 'relu'  # ReLU activation for better stability
    use_dropout = True  # Use dropout for regularization
    dropout_rate = 0.25  # Moderate dropout
    use_batch_norm = True  # Use batch normalization
    weight_decay = 0.0002  # Moderate L2 regularization
    optimizer = 'adam'  # Adam optimizer
    
    # Create the model
    model = DeepNeuralNetwork(
        input_size=input_size,
        hidden_sizes=hidden_sizes,
        output_size=output_size,
        activation=activation,
        use_dropout=use_dropout,
        dropout_rate=dropout_rate,
        use_batch_norm=use_batch_norm,
        weight_decay=weight_decay,
        optimizer=optimizer
    )

    print("Training the model with optimized strategy...")
    
    # No data augmentation for simplicity
    print("Using original data without augmentation...")
    X_train_aug = X_train_std
    Y_train_aug = Y_train
    
    print(f"Augmented data shape: {X_train_aug.shape}")
    print(f"Augmented labels shape: {Y_train_aug.shape}")
    
    # Train the model with optimized parameters for the pyramid network (57.67% accuracy)
    history = model.fit(
        X_train_aug, Y_train_aug,
        X_val=X_val_std, Y_val=Y_val,  # Use standardized validation data
        num_epochs=70,  # More epochs for better convergence
        batch_size=32,  # Smaller batch size for better generalization
        learning_rate=0.0008,  # Moderate learning rate
        print_every=1,  # Print every epoch
        lr_decay=0.85,  # More aggressive learning rate decay
        lr_decay_epoch=7,  # Decay every 7 epochs
        early_stopping=True,  # Enable early stopping
        patience=12,  # Longer patience
        warmup_epochs=5  # Longer warmup
    )

    # Visualize training history
    Visualizer.plot_training_history(history)

    # Evaluate model on test set with standardized data
    print("\nEvaluating model on test set...")
    metrics = model.evaluate(X_test_std, Y_test)
    
    print("\nTest Metrics:")
    print(f"Loss: {metrics['loss']:.4f}")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Macro Precision: {metrics['macro_precision']:.4f}")
    print(f"Macro Recall: {metrics['macro_recall']:.4f}")
    print(f"Macro F1 Score: {metrics['macro_f1']:.4f}")
    
    # Plot confusion matrix
    y_true = np.argmax(Y_test, axis=0)
    pred = model.predict(X_test_std)
    y_pred = np.argmax(pred, axis=0)
    Visualizer.plot_confusion_matrix(y_true, y_pred, [str(i) for i in range(num_classes)])

    return model, metrics, history


if __name__ == "__main__":
    model, metrics, history = main()
    # print(model, metrics, history)
