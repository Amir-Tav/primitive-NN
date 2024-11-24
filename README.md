# primitive Neural Network (NN From Scratch)

Welcome to the **Neural Network From Scratch** project! In this project, we built a simple, yet powerful neural network from the ground up, without relying on libraries like TensorFlow or Keras. Instead, we used **Numpy** and **linear algebra** to understand the raw mechanics behind neural networks. Let's dive in! 🤖💡

## Project Overview
The goal of this project was to implement a basic neural network with 3 layers:
1. **Input Layer**: 784 nodes corresponding to the pixels in the 28x28 MNIST images.
2. **Hidden Layer**: 10 nodes, which helps in learning complex patterns.
3. **Output Layer**: 10 units, one for each digit (0-9) that the model is classifying.

### Why Build a Neural Network From Scratch?
Building a neural network from scratch is not only fun, but it also gives you a deeper understanding of the algorithms behind machine learning. Instead of relying on pre-built frameworks, we manually implement key components like forward propagation, backward propagation, and activation functions, including **ReLU** and **Softmax**.

### The Dataset
For this project, we used the **MNIST** dataset, which consists of **28x28 grayscale images** of handwritten digits. Our task was to build a model that can classify these digits based on the pixel values. This makes it a **classification problem**.

---

## Key Concepts and Code Implementation ⚙️

Here’s a brief overview of the important parts of the code:

### 1. **Data Preprocessing** 
Before training, the data is shuffled and split into training and development sets. The pixel values are also normalized to a range between 0 and 1.

```python

data = np.array(data)
m, n = data.shape
np.random.shuffle(data)  # Shuffle before splitting

# Development set (1000 samples)
data_dev = data[0:1000].T
Y_dev = data_dev[0]
X_dev = data_dev[1:n]
X_dev = X_dev / 255.  # Normalize

# Training set (remaining samples)
data_train = data[1000:m].T
Y_train = data_train[0]
X_train = data_train[1:n]
X_train = X_train / 255.  # Normalize
```


---

### 2. **Network Initialization**
We initialize the weights and biases for the neural network using random values. This is where the magic starts! 

```python

def init_params():
    W1 = np.random.rand(10, 784) - 0.5  # Weights for layer 1
    b1 = np.random.rand(10, 1) - 0.5  # Bias for layer 1
    W2 = np.random.rand(10, 10) - 0.5  # Weights for layer 2
    b2 = np.random.rand(10, 1) - 0.5  # Bias for layer 2
    return W1, b1, W2, b2

```

---

### 3. **Forward Propagation**
We calculate activations at each layer to determine the output of the network. The ReLU activation function is applied to the hidden layer, and Softmax is used at the output layer to produce probabilities.

```python 

def forward_prop(W1, b1, W2, b2, X):
    Z1 = W1.dot(X) + b1  # Weighted sum for hidden layer
    A1 = ReLU(Z1)  # Apply ReLU activation
    Z2 = W2.dot(A1) + b2  # Weighted sum for output layer
    A2 = softmax(Z2)  # Apply Softmax to get probabilities
    return Z1, A1, Z2, A2

```

---

### 4. **Backward Propagation**
We compute the gradients for each weight and bias, helping the network adjust during training. This step allows the model to learn from its errors!

```python

def backward_prop(Z1, A1, Z2, A2, W1, W2, X, Y):
    one_hot_Y = one_hot(Y)  # One-hot encode the labels
    dZ2 = A2 - one_hot_Y  # Error at output layer
    dW2 = 1 / m * dZ2.dot(A1.T)  # Gradients for W2
    db2 = 1 / m * np.sum(dZ2)  # Gradients for b2
    dZ1 = W2.T.dot(dZ2) * ReLU_deriv(Z1)  # Error at hidden layer
    dW1 = 1 / m * dZ1.dot(X.T)  # Gradients for W1
    db1 = 1 / m * np.sum(dZ1)  # Gradients for b1
    return dW1, db1, dW2, db2

```

---

### 5. Training the Network**
We use gradient descent to minimize the loss and optimize the network's weights and biases over multiple iterations.

```python  

def gradient_descent(X, Y, alpha, iterations):
    W1, b1, W2, b2 = init_params()
    for i in range(iterations):
        Z1, A1, Z2, A2 = forward_prop(W1, b1, W2, b2, X)
        dW1, db1, dW2, db2 = backward_prop(Z1, A1, Z2, A2, W1, W2, X, Y)
        W1, b1, W2, b2 = update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha)
        if i % 10 == 0:
            print("Iteration:", i)
            predictions = get_predictions(A2)
            print(get_accuracy(predictions, Y))
    return W1, b1, W2, b2


```

---

### 6. Results
After training the model for 500 epochs with a learning rate of 0.1, we were able to achieve an average accuracy of 86%. Not bad for a simple neural network trained from scratch!

* **Testing the Model:**
We tested the model by making predictions on a few images. The model successfully predicted 3 out of 4 images correctly, showing that it has learned useful patterns from the data.

```python 

def make_predictions(X, W1, b1, W2, b2):
    _, _, _, A2 = forward_prop(W1, b1, W2, b2, X)
    predictions = get_predictions(A2)
    return predictions

def test_prediction(index, W1, b1, W2, b2):
    current_image = X_train[:, index, None]
    prediction = make_predictions(X_train[:, index, None], W1, b1, W2, b2)
    label = Y_train[index]
    print("Prediction:", prediction)
    print("Label:", label)
    
    current_image = current_image.reshape((28, 28)) * 255
    plt.gray()
    plt.imshow(current_image, interpolation='nearest')
    plt.show()


```

---

### 7. Conclusion
This project has been an exciting journey of understanding how neural networks function at a fundamental level. We were able to create a basic neural network, train it on the MNIST dataset, and achieve an accuracy of 86%.

**Future Work**
* Fine-tune the model by adjusting the learning rate or adding dynamic learning rates.
* Add more hidden layers to improve the model’s learning capacity.
* Train the model for more epochs to achieve better performance.

**Objective**
The main objective of this project was to gain a deeper understanding of neural networks, and this knowledge will be incredibly useful in more advanced machine learning tasks.

**Credits** 
A big thank you to **Samson Zhang** for his tutorial that helped me understand how neural networks work from scratch. If you're interested, you can watch his full video [here](https://www.youtube.com/watch?v=w8yWXqWQYmU).
