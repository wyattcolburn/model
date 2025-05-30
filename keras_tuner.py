import tensorflow as tf
from tensorflow import keras
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def manual_hyperparameter_search(X_train, y_train, X_val, y_val):
    """Simple manual hyperparameter search"""
    best_loss = float('inf')
    best_params = None
    best_model = None
    
    # Define parameter combinations to try
    param_combinations = [
        {'neurons': 128, 'layers': 3, 'dropout': 0.3, 'lr': 0.001},
        {'neurons': 256, 'layers': 2, 'dropout': 0.2, 'lr': 0.01}, 
        {'neurons': 64, 'layers': 4, 'dropout': 0.4, 'lr': 0.001},
        {'neurons': 512, 'layers': 2, 'dropout': 0.5, 'lr': 0.0001},
        {'neurons': 128, 'layers': 2, 'dropout': 0.3, 'lr': 0.01},
        {'neurons': 256, 'layers': 3, 'dropout': 0.4, 'lr': 0.001},
    ]
    
    print(f"Testing {len(param_combinations)} different configurations...")
    
    for i, params in enumerate(param_combinations):
        print(f"\nConfiguration {i+1}/{len(param_combinations)}: {params}")
        
        # Build model with current parameters
        model = keras.Sequential()
        
        # Add hidden layers
        for layer in range(params['layers']):
            model.add(keras.layers.Dense(params['neurons'], activation='relu'))
            model.add(keras.layers.Dropout(params['dropout']))
        
        # Output layer (2 outputs for cmd_vel: linear and angular)
        model.add(keras.layers.Dense(2))  # Adjust based on your labels
        
        # Compile model
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=params['lr']),
            loss='mse',
            metrics=['mae']
        )
        
        # Train model
        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=30,
            batch_size=32,
            verbose=0,  # Silent training
            callbacks=[keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True)]
        )
        
        # Get best validation loss
        val_loss = min(history.history['val_loss'])
        
        # Check if this is the best model so far
        if val_loss < best_loss:
            best_loss = val_loss
            best_params = params.copy()
            best_model = model
            print(f"  ✓ NEW BEST! Val loss: {val_loss:.4f}")
        else:
            print(f"  Val loss: {val_loss:.4f}")
    
    print(f"Parameters: {best_params}")
    print(f"Best validation loss: {best_loss:.4f}")
    
    return best_model, best_params

def run_manual_tuning():
    """Main function to run the manual tuning"""
    
    # 1. Load your data
    print("Loading data...")
    features = pd.read_csv("combine_dkr/combined_features.csv")
    labels = pd.read_csv("combine_dkr/combined_labels.csv")
    
    print(f"Features shape: {features.shape}")
    print(f"Labels shape: {labels.shape}")
    
    # 2. Prepare data
    X = features.values
    y = labels.values
    
    # 3. Split into train/validation/test
    X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.25, random_state=42)
    
    print(f"Train set: {X_train.shape}")
    print(f"Validation set: {X_val.shape}")
    print(f"Test set: {X_test.shape}")
    
    # 4. Scale the data
    print("Scaling data...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    
    # 5. Run hyperparameter search
    best_model, best_params = manual_hyperparameter_search(
        X_train_scaled, y_train, X_val_scaled, y_val
    )
    
    # 6. Evaluate on test set
    print(f"\nEvaluating best model on test set...")
    test_loss, test_mae = best_model.evaluate(X_test_scaled, y_test, verbose=0)
    print(f"Test Loss (MSE): {test_loss:.4f}")
    print(f"Test MAE: {test_mae:.4f}")
    
    # 7. Save the best model
    best_model.save('best_manual_tuned_model.h5')
    print(f"Best model saved as 'best_manual_tuned_model.h5'")
    
    # 8. Save the scaler for later use
    import joblib
    joblib.dump(scaler, 'scaler.pkl')
    print("Scaler saved as 'scaler.pkl'")
    
    return best_model, best_params

if __name__ == "__main__":
    best_model, best_params = run_manual_tuning()
