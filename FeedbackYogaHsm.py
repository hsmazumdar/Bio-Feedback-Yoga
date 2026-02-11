"""
================================================================================
FeedbackYogaHsm.py
================================================================================
Author:   Himanshu S Mazumdar
Date:     February 11, 2025
Objective: Simulate EEG-based meditation feedback using Hilbert space-filling
           curve mapping. Classifies thinking vs. meditating states via spectral
           band powers (Delta, Theta, Alpha, Beta), maps cognitive states to 1D
           Hilbert coordinates, trains an MLP classifier, and provides simulated
           closed-loop feedback for yoga/meditation practice.
================================================================================
"""

# PSEUDO-CODE STRUCTURE FOR THE SIMULATION
import numpy as np
import pandas as pd
from scipy import signal, stats
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt

# === 1. DATA LOADING & PREPROCESSING ===
# Load EEG data (e.g., from CSV). Assume columns: ['channel_1', ..., 'channel_n', 'label']
# label = 0 (thinking), 1 (meditating)
import os
_data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'eeg_meditation_data.csv')
if os.path.isfile(_data_path):
    data = pd.read_csv(_data_path)
    raw_eeg = data.iloc[:, :-1].values
    labels = data['label'].values
else:
    # Synthetic data fallback (Paper Section 2.1: E(t)=[A,B,D,Theta], fs typical 256 Hz)
    np.random.seed(42)
    n_epochs, n_samples = 200, 512  # 2 s at 256 Hz per epoch
    raw_eeg = np.random.randn(n_epochs, n_samples) * 0.5
    labels = np.random.randint(0, 2, n_epochs)
    print("[Demo] Using synthetic EEG data (no eeg_meditation_data.csv found).")

# === 2. SPECTRAL DECOMPOSITION (Paper's Section 2.1) ===
# For each epoch (e.g., 2-second window), compute band powers
fs = 256  # Sampling frequency example
band_defs = {'Delta': (0.5, 4), 'Theta': (4, 8), 'Alpha': (8, 12), 'Beta': (13, 30)}
band_powers = []

for epoch in raw_eeg:
    freqs, psd = signal.welch(epoch, fs, nperseg=fs*2)
    epoch_bands = []
    for band, (low, high) in band_defs.items():
        idx = np.logical_and(freqs >= low, freqs <= high)
        epoch_bands.append(np.log(np.sum(psd[idx]) + 1e-10))  # Log-power
    band_powers.append(epoch_bands)

band_powers = np.array(band_powers)  # Shape: [n_epochs, 4_bands]

# Normalize per subject/session (simplified per-epoch Z-score)
scaler = MinMaxScaler()
band_powers_norm = scaler.fit_transform(band_powers)

# === 3. HILBERT SPACE-FILLING CURVE MAPPING (Paper's Section 2.2) ===
# We implement a simplified version. For a true Hilbert curve, use a library like 'hilbertcurve'
def map_to_hilbert_1d(point_4d):
    """Maps a normalized 4D point [A, B, D, T] to an approximate 1D Hilbert index."""
    # Simplified for demonstration: weighted sum with bit-interleaving logic
    # A more rigorous implementation uses a Hilbert curve library.
    # For proof-of-concept, we use a locality-sensitive hash.
    point_int = (point_4d * 1023).astype(int)  # Scale to integer grid
    h = 0
    for i in range(10):  # 10 bits of precision
        h |= ((point_int[0] >> i) & 1) << (4*i + 0)
        h |= ((point_int[1] >> i) & 1) << (4*i + 1)
        h |= ((point_int[2] >> i) & 1) << (4*i + 2)
        h |= ((point_int[3] >> i) & 1) << (4*i + 3)
    return h

hilbert_coords = np.array([map_to_hilbert_1d(p) for p in band_powers_norm])
hilbert_coords = MinMaxScaler().fit_transform(hilbert_coords.reshape(-1, 1)).flatten()

# === 4. NEURAL NETWORK CLASSIFICATION (Paper's Section 2.3) ===
# Use Hilbert coordinate + optionally raw bands as input
X = np.column_stack([hilbert_coords, band_powers_norm])  # Input features
y = labels

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a simple feedforward network
nn = MLPClassifier(hidden_layer_sizes=(64, 32), activation='relu', max_iter=500, random_state=42)
nn.fit(X_train, y_train)

# Predict and evaluate
y_pred = nn.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Neural Network Classification Accuracy: {accuracy:.2%}")

# === 5. VISUALIZATION & RESULTS (Aligns with Paper's Figs 2 & 3) ===
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

# Plot 1: Hilbert Coordinate Distribution by State
axes[0].hist(hilbert_coords[labels==0], alpha=0.7, label='Thinking (High Beta)', bins=30, color='red')
axes[0].hist(hilbert_coords[labels==1], alpha=0.7, label='Meditating (High Alpha/Delta)', bins=30, color='blue')
axes[0].set_xlabel('Hilbert Cognitive Coordinate h(t)')
axes[0].set_ylabel('Frequency')
axes[0].set_title('Hilbert Mapping Separates Mental States')
axes[0].legend()
axes[0].grid(True, linestyle='--', alpha=0.6)

# Plot 2: 2D Projection of EEG State Space (Alpha vs. Beta) — Paper Fig. 3 style
# band_powers_norm columns: [Delta, Theta, Alpha, Beta] -> index 2=Alpha, 3=Beta
axes[1].scatter(band_powers_norm[labels==0, 2], band_powers_norm[labels==0, 3], alpha=0.6, label='Thinking', c='red', s=10)
axes[1].scatter(band_powers_norm[labels==1, 2], band_powers_norm[labels==1, 3], alpha=0.6, label='Meditating', c='blue', s=10)
axes[1].set_xlabel('Normalized Alpha Power')
axes[1].set_ylabel('Normalized Beta Power')
axes[1].set_title('Raw EEG Band-Power Feature Space (Alpha vs Beta)')
axes[1].legend()
axes[1].grid(True, linestyle='--', alpha=0.6)

# Plot 3: Trajectory in Hilbert Space over time (for a sample session)
sample_session_len = 100
axes[2].plot(range(sample_session_len), hilbert_coords[:sample_session_len], marker='o', linestyle='-', linewidth=1, markersize=3)
axes[2].fill_between(range(sample_session_len), 0, 1, where=(labels[:sample_session_len]==1), color='blue', alpha=0.2, label='Meditation Block')
axes[2].set_xlabel('Time Epoch')
axes[2].set_ylabel('Hilbert Coordinate h(t)')
axes[2].set_title('Cognitive Trajectory: Movement to Meditative State')
axes[2].legend()
axes[2].grid(True, linestyle='--', alpha=0.6)

plt.tight_layout()
if __import__('matplotlib').get_backend().lower() == 'agg':
    plt.savefig('YogaIITMandi_figures.png', dpi=150, bbox_inches='tight')
plt.show()

# === 6. SIMULATED CLOSED-LOOP FEEDBACK (Conceptual) ===
print("\n--- Simulated Closed-Loop Result ---")
# Simulate a simple feedback rule: if h(t) > threshold and Beta is low, provide positive feedback.
threshold = 0.7
beta_threshold = 0.3
for i in range(5):  # Simulate 5 epochs
    h = hilbert_coords[i]
    beta = band_powers_norm[i, 3]
    if h > threshold and beta < beta_threshold:
        print(f"Epoch {i}: h(t)={h:.2f}, Beta={beta:.2f} -> FEEDBACK: 'Good meditation state. Maintain focus.'")
    else:
        print(f"Epoch {i}: h(t)={h:.2f}, Beta={beta:.2f} -> FEEDBACK: 'Adjust breathing, relax thoughts.'")