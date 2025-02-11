# Import required libraries
import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tensorflow.keras.optimizers import Adagrad, RMSprop, Adam, SGD

print("UID: 22BAI71060")

# 1. Data Preparation
print("\n=== Data Preparation ===")

# Load the MNIST dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Print initial data shapes
print("Initial shapes:")
print("Training data shape:", x_train.shape)
print("Training labels shape:", y_train.shape)
print("Testing data shape:", x_test.shape)
print("Testing labels shape:", y_test.shape)

# Normalize the images to range [0, 1]
x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255

# Flatten the images (reshape to 1D)
x_train = x_train.reshape((x_train.shape[0], 28 * 28))
x_test = x_test.reshape((x_test.shape[0], 28 * 28))

# One-hot encode the labels
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

print("\nAfter preprocessing:")
print("Training data shape:", x_train.shape)
print("Training labels shape:", y_train.shape)

# 2. Model Definition
print("\n=== Model Creation ===")

# Define optimizer classes and parameters (not instances)
optimizers = {
    'Adagrad': (Adagrad, {'learning_rate': 0.01}),
    'RMSprop': (RMSprop, {'learning_rate': 0.001}),
    'Adam': (Adam, {'learning_rate': 0.001}),
    'SGD': (SGD, {'learning_rate': 0.01})
}

epochs_list = [5, 10, 15]
batch_sizes = [32, 64, 128]

# Create empty lists to store results
results = []

print("\n=== Comparative Analysis of Optimizers ===")

for opt_name, (optimizer_class, optimizer_params) in optimizers.items():
    for epoch in epochs_list:
        for batch_size in batch_sizes:
            print(f"\nTraining with: {opt_name}, Epochs: {epoch}, Batch Size: {batch_size}")
            
            # Reset model
            model = Sequential([
                Dense(512, activation='relu', input_shape=(28 * 28,)),
                Dense(512, activation='relu'),
                Dense(256, activation='relu'),
                Dense(128, activation='relu'),
                Dense(64, activation='relu'),
                Dense(10, activation='softmax')
            ])
            
            # Create a fresh optimizer instance for each model
            optimizer = optimizer_class(**optimizer_params)
            
            # Compile model
            model.compile(optimizer=optimizer,
                        loss='categorical_crossentropy',
                        metrics=['accuracy'])
            
            # Train model
            history = model.fit(x_train, y_train,
                              epochs=epoch,
                              batch_size=batch_size,
                              validation_split=0.2,
                              verbose=1)
            
            # Evaluate model
            test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=1)
            
            # Store results
            results.append({
                'Optimizer': opt_name,
                'Epochs': epoch,
                'Batch Size': batch_size,
                'Test Accuracy': test_accuracy,
                'Test Loss': test_loss,
                'Final Training Accuracy': history.history['accuracy'][-1],
                'Final Validation Accuracy': history.history['val_accuracy'][-1]
            })

# Convert results to DataFrame
results_df = pd.DataFrame(results)

# Display results table
print("\n=== Results Table ===")
print(results_df.to_string(index=False))

# Create visualization
plt.figure(figsize=(15, 10))

# Plot test accuracy comparison
plt.subplot(2, 1, 1)
for opt in optimizers.keys():
    opt_results = results_df[results_df['Optimizer'] == opt]
    plt.plot(opt_results['Epochs'], opt_results['Test Accuracy'], 'o-', label=opt)

plt.title('Optimizer Performance Comparison')
plt.xlabel('Epochs')
plt.ylabel('Test Accuracy')
plt.legend()
plt.grid(True)

# Plot training vs validation accuracy for AdaGrad
plt.subplot(2, 1, 2)
adagrad_results = results_df[results_df['Optimizer'] == 'Adagrad']
plt.plot(adagrad_results['Epochs'], adagrad_results['Final Training Accuracy'], 'o-', label='Training Accuracy')
plt.plot(adagrad_results['Epochs'], adagrad_results['Final Validation Accuracy'], 'o-', label='Validation Accuracy')
plt.title('AdaGrad Performance Analysis')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig('optimizer_comparison.png')
plt.show()

# Save results to CSV
results_df.to_csv('optimizer_results.csv', index=False)
print("\nResults have been saved to 'optimizer_results.csv'")
print("Visualization has been saved as 'optimizer_comparison.png'")