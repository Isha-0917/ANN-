import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import confusion_matrix, classification_report
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, Conv2DTranspose

# Function to train and evaluate the CNN Autoencoder
def cnn_autoencoder(X_train, X_test, y_test, epochs=50, batch_size=16):
    # Build CNN Autoencoder
    input_img = Input(shape=(6, 5, 1))

    # Encoder
    x = Conv2D(16, (3, 3), activation='relu', padding='same')(input_img)
    x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)

    # Decoder
    x = Conv2DTranspose(8, (3, 3), activation='relu', padding='same')(x)
    decoded = Conv2DTranspose(1, (3, 3), activation='sigmoid', padding='same')(x)

    autoencoder = Model(input_img, decoded)
    autoencoder.compile(optimizer='adam', loss='mse')

    # Train the model
    history = autoencoder.fit(X_train, X_train, epochs=epochs, batch_size=batch_size, validation_split=0.2, verbose=0)

    # Predict and calculate reconstruction error
    X_pred = autoencoder.predict(X_test)
    mse = np.mean(np.power(X_test - X_pred, 2), axis=(1, 2, 3))

    # Set threshold based on training MSE (95th percentile)
    threshold = np.percentile(mse, 95)
    y_pred = [0 if e > threshold else 1 for e in mse]

    return mse, threshold, y_pred, y_test

# Streamlit UI Components
st.title('CNN-based Anomaly Detection using Autoencoder')

# Load the dataset (Breast Cancer dataset)
st.sidebar.header("Dataset Info")
st.write("Using the Breast Cancer dataset (from sklearn) for anomaly detection.")

# Step 1: Load and process the data
data = load_breast_cancer()
X = data.data
y = data.target  # 0 = anomaly (malignant), 1 = normal (benign)

# Step 2: Normalize data
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# Step 3: Reshape tabular data into "image-like" input for CNN (6x5x1)
X_reshaped = X_scaled.reshape(-1, 6, 5, 1)

# Step 4: Split data
X_train = X_reshaped[y == 1]   # Train only on normal samples
X_test = X_reshaped
y_test = y

# Step 5: Run CNN Autoencoder
mse, threshold, y_pred, y_test = cnn_autoencoder(X_train, X_test, y_test)

# Show results
st.subheader("Reconstruction Error Threshold")
st.write(f"Threshold for anomaly detection: {threshold}")

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
st.subheader("Confusion Matrix")
st.write(cm)

# Classification report
st.subheader("Classification Report")
report = classification_report(y_test, y_pred)
st.text(report)

# Reconstruction error histogram
st.subheader("Reconstruction Error Distribution")
plt.figure(figsize=(10, 6))
plt.hist(mse, bins=50, color='blue', alpha=0.7)
plt.axvline(threshold, color='red', linestyle='--', label='Threshold')
plt.title("Reconstruction Error")
plt.xlabel("MSE")
plt.ylabel("Frequency")
plt.legend(loc="upper right")
st.pyplot()
