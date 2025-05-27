import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler


def analyze_scaling_impact(X_data, feature_names=None):
    """
    Analyze the impact of different scaling methods on features
    """
    if feature_names is None:
        feature_names = [f'Feature_{i}' for i in range(X_data.shape[1])]

    scalers = {
        'StandardScaler': StandardScaler(),
        'MinMaxScaler': MinMaxScaler(),
        'RobustScaler': RobustScaler()
    }

    # Convert to DataFrame for easier handling
    if isinstance(X_data, np.ndarray):
        df_features = pd.DataFrame(X_data, columns=feature_names)
    else:
        df_features = X_data.copy()
        feature_names = df_features.columns.tolist()

    n_features = len(feature_names)
    fig, axes = plt.subplots(len(scalers) + 1, n_features, figsize=(4*n_features, 16))

    # Handle single feature case
    if n_features == 1:
        axes = axes.reshape(-1, 1)

    # Original distributions
    for i, feature in enumerate(feature_names):
        axes[0, i].hist(df_features[feature].dropna(), bins=50, alpha=0.7)
        axes[0, i].set_title(f'Original {feature}')
        axes[0, i].set_ylabel('Frequency')

    # Scaled distributions
    for scaler_idx, (scaler_name, scaler) in enumerate(scalers.items()):
        scaled_data = scaler.fit_transform(df_features.dropna())
        scaled_df = pd.DataFrame(scaled_data, columns=feature_names)

        for i, feature in enumerate(feature_names):
            axes[scaler_idx + 1, i].hist(scaled_df[feature], bins=50, alpha=0.7)
            axes[scaler_idx + 1, i].set_title(f'{scaler_name} {feature}')
            if i == 0:
                axes[scaler_idx + 1, i].set_ylabel('Frequency')

    plt.tight_layout()
    plt.show()

    # Statistics comparison
    print("=== SCALING STATISTICS COMPARISON ===")
    original_stats = df_features.describe()
    print("Original Data:")
    print(original_stats)

    for scaler_name, scaler in scalers.items():
        scaled_data = scaler.fit_transform(df_features.dropna())
        scaled_df = pd.DataFrame(scaled_data, columns=feature_names)
        print(f"\n{scaler_name}:")
        print(scaled_df.describe())


def scale_train_data(X_train, y_train, scaler_type='robust'):
    """
    Scale training data and return scalers for later use on test data

    Parameters:
    X_train: Training features (DataFrame or numpy array)
    y_train: Training target (Series or numpy array)
    scaler_type: 'standard', 'minmax', or 'robust'

    Returns:
    X_train_scaled, y_train_scaled, x_scaler, y_scaler
    """

    # Choose scaler type
    scaler_options = {
        'standard': StandardScaler(),
        'minmax': MinMaxScaler(),
        'robust': RobustScaler()
    }

    x_scaler = scaler_options[scaler_type]
    y_scaler = StandardScaler()  # Usually StandardScaler for target

    # Scale features
    X_train_scaled = x_scaler.fit_transform(X_train)

    # Handle target scaling - convert to numpy array first if it's a pandas Series
    if hasattr(y_train, 'values'):
        # It's a pandas Series or DataFrame
        y_train_array = y_train.values
    else:
        # It's already a numpy array
        y_train_array = y_train

    # Scale target (reshape to 2D for scaler)
    if y_train_array.ndim == 1:
        y_train_reshaped = y_train_array.reshape(-1, 1)
    else:
        y_train_reshaped = y_train_array

    y_train_scaled = y_scaler.fit_transform(y_train_reshaped)

    # If original y_train was 1D, flatten the scaled version
    if y_train_array.ndim == 1:
        y_train_scaled = y_train_scaled.flatten()

    print(f"=== SCALING APPLIED ===")
    print(f"X_scaler: {type(x_scaler).__name__}")
    print(f"y_scaler: {type(y_scaler).__name__}")
    print(f"X_train shape: {X_train.shape} -> {X_train_scaled.shape}")
    print(f"y_train shape: {y_train.shape} -> {y_train_scaled.shape}")

    return X_train_scaled, y_train_scaled, x_scaler, y_scaler


# Function to scale test data using fitted scalers
def scale_test_data(X_test, y_test, x_scaler, y_scaler):
    """
    Scale test data using pre-fitted scalers
    """
    X_test_scaled = x_scaler.transform(X_test)

    if y_test.ndim == 1:
        y_test_reshaped = y_test.reshape(-1, 1)
    else:
        y_test_reshaped = y_test

    y_test_scaled = y_scaler.transform(y_test_reshaped)

    if y_test.ndim == 1:
        y_test_scaled = y_test_scaled.flatten()

    return X_test_scaled, y_test_scaled


# Function to inverse transform predictions
def inverse_transform_predictions(y_pred_scaled, y_scaler):
    """
    Convert scaled predictions back to original scale
    """
    if y_pred_scaled.ndim == 1:
        y_pred_reshaped = y_pred_scaled.reshape(-1, 1)
    else:
        y_pred_reshaped = y_pred_scaled

    y_pred_original = y_scaler.inverse_transform(y_pred_reshaped)

    if y_pred_scaled.ndim == 1:
        y_pred_original = y_pred_original.flatten()

    return y_pred_original
