"""
MNSIT : Madified National Institute of Standard and Technology
Dataset of handwritten digits (0-9)
    60,000 training images
    10,000 test images
    28X28 pixel grayscale images
    10 classes (digits 0-9)
"""

import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)

n_train = 5000
n_test = 1000
img_size = 28

template = (np.random.rand(10, img_size, img_size) > 0.88).astype(float) * 100

def make_image(labels):
    noise = np.clip(np.random.rand(len(labels), img_size, img_size) * 40 +30, 0, 255)

    return np.clip(noise + template[labels], 0, 255)

y_train = np.repeat(np.arange(10), n_train//10)
np.random.shuffle(y_train)
y_test = np.repeat(np.arange(10), n_test//10)
np.random.shuffle(y_test)
X_train = make_image(y_train)
X_test = make_image(y_test)

print(f"Created MNIST-like dataset:")
print(f"  Training: {X_train.shape} (5000 images, 28X28)")
print(f"  Testing: {X_test.shape} (1000 images, 28X28)")
print(f"  Classes: 10 (digits 0-9)")

"""
NOTE: this is NOT real MNIST. Each 'digit' is random noise plus a fixed random pattern for
that class, so the code runs without a download. Real MNIST: 60,000 train / 10,000 test.
Real MNIST on your computer (needs internet), two standard ways:
    from tensorflow.keras.datasets import mnist
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
  or
    from sklearn.datasets import fetch_openml
    X, y = fetch_openml('mnist_784', version=1, return_X_y=True, as_frame=False)
"""

# Sanity Check : 
class_means = np.array([
    X_train[y_train == d].mean(axis=0).ravel() 
    for d in range(10)
    ])

dist = (
    (X_test.reshape(len(X_test), 1, -1)
      - class_means[None, :, :]) ** 2
      ).sum(axis=2)

sanity_acc = np.mean(np.argmin(dist, axis=1) == y_test)
print(f"Sanity check - nearest class-average classifier: {sanity_acc*100:.1f}% test accuracy (chance = 10%)")

ig, axes = plt.subplots(2, 5, figsize=(15, 6))
for digit in range(10):
    idx = np.where(y_train == digit)[0][0]
    ax = axes.flat[digit]
    ax.imshow(X_train[idx], cmap='gray')
    ax.set_title(f'Digit {digit}')
    ax.axis('off')
 
plt.suptitle('SYNTHETIC samples (noise + class pattern) - not real handwriting', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('mnist_samples.png', dpi=300, bbox_inches='tight')
print("✓ Saved: mnist_samples.png")
plt.show()
 
print("\n4. DATA STATISTICS")
print("-"*70)
 
for digit in range(10):
    count = np.sum(y_train == digit)
    print(f"  Digit {digit}: {count} images")


# data statistics  

for digit in range(10):
    count = np.sum(y_train == digit)
    print(f" Digit {digit} : {count} image")

print("Normalize Image")

X_train_norm = X_train / 255.0
X_test_norm = X_test / 255.0

print(f"Before: {X_train.min():.0f}-{X_train.max():.0f}")
print(f"After: {X_train_norm.min():.3f}-{X_train_norm.max():.3f}")

print(" Flatten for Neural Network ")

X_train_flat = X_train_norm.reshape(len(X_train_norm), -1)
Y_train_flat = X_test_norm.reshape(len(X_test_norm), -1)

print(f"Original shape: {X_train_norm.shape}")
print(f"Flattened: {X_train_flat.shape}")

np.save('mnist_X_train.npy', X_train_norm)
np.save('mnist_X_test.npy', X_test_norm)
np.save('mnist_y_train.npy', y_train)
np.save('mnist_y_test.npy', y_test)
print(X_train.shape)
print(y_train.shape)
print(X_train.min(), X_train.max())