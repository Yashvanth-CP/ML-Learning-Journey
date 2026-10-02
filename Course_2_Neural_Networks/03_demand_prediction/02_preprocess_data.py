"""
PROJECT 1: DEMAND PREDICTION
Step 2: Prepare & Normalize Data
"""

import numpy as np
import pandas as pd 
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

print("DEMAND PREDICTION: Step 2 - Prepare Data")

print("\n WHY PREPARE DATA?")
print("_-"*20)

"""
Neural networks work best when:
  ✓ Features are normalized (mean≈0, std≈1)
  ✓ Data is split into train/test
  ✓ No missing values
  ✓ Features are on same scale
 
Raw data problems:
  ✗ Features have different scales
    - Marketing: 0-1000
    - Day of week: 0-6
    - Temperature: 5-35
  ✗ Large scale differences → training problems
  ✗ Network can't compare importance
 
Solution: Normalize!
"""
X = np.load('coffee_demand_X.npy')
y = np.load('coffee_demand_y.npy')

print(f"Data loaded!")
print(f"  X shape: {X.shape} (samples, features)")
print(f"  y shape: {y.shape} (samples,)")
 
print(f"\nBefore normalization:")
print(f"  X min: {X.min(axis=0)}")
print(f"  X max: {X.max(axis=0)}")
print(f"  y min: {y.min():.2f}")
print(f"  y max: {y.max():.2f}")

print("\n3. NORMALIZE FEATURES")

"""
Standardization (Z-score normalization):
  X_norm = (X - mean(X)) / std(X)
"""

scaler_X = StandardScaler()
X_scaled = scaler_X.fit_transform(X)

print(f"Feature after normalizaton : ")
print(f" X min : {X_scaled.min(axis=0)}")
print(f" X max: {X_scaled.max(axis=0)}")
print(f" X mean : {X_scaled.mean(axis=0)}")
print(f" X std : {X_scaled.std(axis=0)}")

# NOrmalize target 

scaler_y = StandardScaler()
y_scaled= scaler_y.fit_transform(y.reshape(-1, 1)).flatten()

print(f"\nTarget after normalization:")
print(f"  y min: {y_scaled.min():.2f}")
print(f"  y max: {y_scaled.max():.2f}")
print(f"  y mean: {y_scaled.mean():.4f}")
print(f"  y std: {y_scaled.std():.4f}")

print(" Train test spilt ")

# split data 

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_scaled, test_size=0.2, random_state=42)

print(f"Train set: {len(X_train)} samples ({len(X_train)/len(X)*100:.0f}%)")
print(f"Test set:  {len(X_test)} samples ({len(X_test)/len(X)*100:.0f}%)")
print(f"\nTrain shapes:")
print(f"  X_train: {X_train.shape}")
print(f"  y_train: {y_train.shape}")
print(f"\nTest shapes:")
print(f"  X_test: {X_test.shape}")
print(f"  y_test: {y_test.shape}")

print("\n5. DATA DISTRIBUTION")
print("-"*70)
 
import matplotlib.pyplot as plt
 
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
 
# Train vs Test distribution
ax = axes[0, 0]
ax.hist(y_train, bins=20, alpha=0.5, label='Train', color='blue')
ax.hist(y_test, bins=20, alpha=0.5, label='Test', color='red')
ax.set_xlabel('Revenue (normalized)')
ax.set_ylabel('Frequency')
ax.set_title('Train vs Test Distribution', fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3)
 
# Feature distributions (first 3 features)
features = ['Day of week', 'Temperature', 'Marketing']
for idx, feature_name in enumerate(features):
    ax = axes.flat[idx + 1]
    ax.hist(X_train[:, idx], bins=20, alpha=0.5, label='Train', color='blue')
    ax.hist(X_test[:, idx], bins=20, alpha=0.5, label='Test', color='red')
    ax.set_xlabel(feature_name + ' (normalized)')
    ax.set_ylabel('Frequency')
    ax.set_title(f'{feature_name} Distribution', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
 
plt.tight_layout()
plt.savefig('data_split_analysis.png', dpi=300, bbox_inches='tight')
print("\n✓ Saved: data_split_analysis.png")
plt.show()
 
 
print("\n6. SAVE PREPARED DATA")
print("-"*70)
 
# Save everything
np.save('X_train.npy', X_train)
np.save('X_test.npy', X_test)
np.save('y_train.npy', y_train)
np.save('y_test.npy', y_test)
 
print("✓ Saved: X_train.npy, X_test.npy, y_train.npy, y_test.npy")


import pickle 
with open('scaler_X.pk1', 'wb') as f :
    pickle.dump(scaler_X, f)

with open('scaler_y.pk1','wb') as f:
    pickle.dump(scaler_y, f)

"""
Data prepared!
 
Actions taken:
  ✓ Loaded raw data ({len(X)} samples)
  ✓ Normalized features using StandardScaler
  ✓ Normalized target using StandardScaler
  ✓ Split into train (80%) and test (20%)
  ✓ Saved scalers for later inference
 
Final dataset:
  Training: {X_train.shape}
  Testing:  {X_test.shape}
  
Key insight:
  Features now have mean≈0, std≈1
  Neural network will train faster & better!
  
Next step: Build and train model!
"""