mport tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import itertools
import json
import os
from datetime import datetime

class ComprehensiveHyperparameterTuner:
    def __init__(self, X_train, y_train, X_val, y_val):
        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val
        self.input_shape = X_train.shape[1]
        self.best_loss = float('inf')
        self.best_params = None
        self.best_model = None
        self.results = []
        
    def create_model(self, params):
        """
        Create model with fixed architecture but tunable hyperparameters
        """
        model = keras.Sequential([
            layers.Input(shape=(self.input_shape,)),
            layers.Dense(256, activation=params['activation'], 
                        kernel_regularizer=keras.regularizers.l2(params['l2_reg'])),
            layers.Dropout(params['dropout']),
            layers.Dense(256, activation=params['activation'],
                        kernel_regularizer=keras.regularizers.l2(params['l2_reg'])),
            layers.Dropout(params['dropout']),
            layers.Dense(256, activation=params['activation'],
                        kernel_regularizer=keras.regularizers.l2(params['l2_reg'])),
            layers.Dropout(params['dropout']),
            layers.Dense(2)  # Output layer with 2 neurons
        ])
        
        # Choose optimizer
        if params['optimizer'] == 'adam':
            optimizer = keras.optimizers.Adam(
                learning_rate=params['learning_rate'],
                beta_1=params['beta_1'],
                beta_2=params['beta_2']
            )
        elif params['optimizer'] == 'rmsprop':
            optimizer = keras.optimizers.RMSprop(
                learning_rate=params['learning_rate'],
                momentum=params['momentum']
            )
        elif params['optimizer'] == 'sgd':
            optimizer = keras.optimizers.SGD(
                learning_rate=params['learning_rate'],
                momentum=params['momentum'],
                nesterov=params['nesterov']
            )
        
        model.compile(
            optimizer=optimizer,
            loss=params['loss_function'],
            metrics=['mae']
        )
        
        return model
    
    def get_callbacks(self, params):
        """Create callbacks based on parameters"""
        callbacks = []
        
        # Early stopping
        callbacks.append(
            keras.callbacks.EarlyStopping(
                patience=params['early_stopping_patience'],
                restore_best_weights=True,
                monitor='val_loss',
                min_delta=params['min_delta']
            )
        )
        
        # Learning rate scheduling
        if params['lr_schedule'] == 'reduce_on_plateau':
            callbacks.append(
                keras.callbacks.ReduceLROnPlateau(
                    factor=params['lr_factor'],
                    patience=params['lr_patience'],
                    min_lr=params['min_lr']
                )
            )
        elif params['lr_schedule'] == 'exponential':
            callbacks.append(
                keras.callbacks.LearningRateScheduler(
                    lambda epoch: params['learning_rate'] * (params['lr_decay'] ** epoch)
                )
            )
        
        return callbacks
    
    def comprehensive_search(self, max_trials=50):
        """
        Comprehensive hyperparameter search with random sampling
        """
        print(f"Starting comprehensive hyperparameter search with {max_trials} trials...")
        
        # Define hyperparameter space
        param_space = {
            'learning_rate': [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05],
            'dropout': [0.1, 0.2, 0.3, 0.4, 0.5],
            'l2_reg': [0.0, 0.001, 0.01, 0.1],
            'activation': ['relu', 'elu', 'swish', 'gelu'],
            'optimizer': ['adam', 'rmsprop', 'sgd'],
            'loss_function': ['mse', 'huber', 'mae'],
            'batch_size': [16, 32, 64, 128],
            'epochs': [50, 100, 150],
            
            # Adam specific
            'beta_1': [0.9, 0.95, 0.99],
            'beta_2': [0.999, 0.9999],
            
            # SGD/RMSprop specific
            'momentum': [0.9, 0.95, 0.99],
            'nesterov': [True, False],
            
            # Learning rate scheduling
            'lr_schedule': ['none', 'reduce_on_plateau', 'exponential'],
            'lr_factor': [0.1, 0.2, 0.5],
            'lr_patience': [5, 10, 15],
            'lr_decay': [0.95, 0.98, 0.99],
            'min_lr': [1e-6, 1e-7, 1e-8],
            
            # Early stopping
            'early_stopping_patience': [10, 15, 20, 25],
            'min_delta': [1e-4, 1e-5, 1e-6],
        }
        
        for trial in range(max_trials):
            print(f"\n{'='*60}")
            print(f"TRIAL {trial + 1}/{max_trials}")
            print(f"{'='*60}")
            
            # Sample parameters randomly
            params = {}
            for param, values in param_space.items():
                params[param] = np.random.choice(values)
            
            try:
                # Create and train model
                model = self.create_model(params)
                callbacks = self.get_callbacks(params)
                
                print(f"Parameters: {json.dumps(convert_numpy_types(params), indent=2)}")
                
                # Train model
                history = model.fit(
                    self.X_train, self.y_train,
                    validation_data=(self.X_val, self.y_val),
                    epochs=params['epochs'],
                    batch_size=params['batch_size'],
                    verbose=1,
                    callbacks=callbacks
                )
                
                # Get best validation loss
                val_loss = min(history.history['val_loss'])
                val_mae = min(history.history['val_mae'])
                final_epoch = len(history.history['val_loss'])
                
                # Store results
                result = {
                    'trial': trial + 1,
                    'params': params.copy(),
                    'val_loss': val_loss,
                    'val_mae': val_mae,
                    'final_epoch': final_epoch,
                    'converged': final_epoch < params['epochs']
                }
                self.results.append(result)
                
                # Check if this is the best model
                if val_loss < self.best_loss:
                    self.best_loss = val_loss
                    self.best_params = params.copy()
                    self.best_model = model
                    print(f"NEW BEST MODEL!")
                    print(f"   Val Loss: {val_loss:.6f}")
                    print(f"   Val MAE: {val_mae:.6f}")
                    print(f"   Converged at epoch: {final_epoch}")
                else:
                    print(f"   Val Loss: {val_loss:.6f} (Best: {self.best_loss:.6f})")
                
            except Exception as e:
                print(f"Trial failed with error: {e}")
                continue
        
        return self.best_model, self.best_params
    
    def grid_search_focused(self):
        """
        Focused grid search around promising areas
        """
        print("Starting focused grid search...")
        
        # Focused parameter combinations based on common good practices

        focused_params = {
            'learning_rate': [0.001, 0.01],
            'dropout': [0.2, 0.3, 0.4],
            'l2_reg': [0.001, 0.01],
            'activation': ['relu', 'swish'],
            'optimizer': ['adam'],
            'loss_function': ['mse', 'huber'],
            'batch_size': [32, 64],
            'beta_1': [0.9],
            'beta_2': [0.999],
            'lr_schedule': ['reduce_on_plateau'],
            'lr_factor': [0.2],
            'lr_patience': [10],
            'min_lr': [1e-6],
            'early_stopping_patience': [15],
            'min_delta': [1e-5],
            'epochs': [100],
        }
        
        # Generate all combinations
        keys = focused_params.keys()
        values = focused_params.values()
        combinations = list(itertools.product(*values))
        
        print(f"Testing {len(combinations)} focused combinations...")
        
        for i, combination in enumerate(combinations):
            params = dict(zip(keys, combination))
            
            # Add default values for unused parameters
            if params['optimizer'] != 'sgd':
                params['momentum'] = 0.9
                params['nesterov'] = False
                
            print(f"\nFocused Trial {i+1}/{len(combinations)}")
            print(f"Params: {params}")
            
            try:
                model = self.create_model(params)
                callbacks = self.get_callbacks(params)
                
                history = model.fit(
                    self.X_train, self.y_train,
                    validation_data=(self.X_val, self.y_val),
                    epochs=params['epochs'],
                    batch_size=params['batch_size'],
                    verbose=0,
                    callbacks=callbacks
                )
                
                val_loss = min(history.history['val_loss'])
                
                if val_loss < self.best_loss:
                    self.best_loss = val_loss
                    self.best_params = params.copy()
                    self.best_model = model
                    print(f"✓ NEW BEST! Val loss: {val_loss:.6f}")
                else:
                    print(f"  Val loss: {val_loss:.6f}")
                    
            except Exception as e:
                print(f" Failed: {e}")
                continue
        
        return self.best_model, self.best_params
    
    def save_results(self, filename=None):
        """Save tuning results to JSON file"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"tuning_results_{timestamp}.json"
        
        results_data = {
            'best_params': self.best_params,
            'best_loss': self.best_loss,
            'all_results': self.results
        }
        
        with open(filename, 'w') as f:
            json.dump(convert_numpy_types(results_data), f, indent=2, default=str)
        
        print(f"Results saved to {filename}")

def convert_numpy_types(obj):
    if isinstance(obj, dict):
        return {k: convert_numpy_types(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(i) for i in obj]
    elif isinstance(obj, (np.integer, np.int64)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64)):
        return float(obj)
    elif isinstance(obj, (np.bool_)):
        return bool(obj)
    else:
        return obj
def run_comprehensive_tuning():
    """Main function to run comprehensive hyperparameter tuning"""
    
    # 1. Load data
    print("Loading data...")
    features = pd.read_csv("combined_dkr_max12_not0/combined_features.csv")
    labels = pd.read_csv("combined_dkr_max12_not0/combined_labels.csv")
    
    print(f"Features shape: {features.shape}")
    print(f"Labels shape: {labels.shape}")
    
    # 2. Prepare data
    X = features.values
    y = labels.values
    
    # 3. Split data
    X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.25, random_state=42)
    
    print(f"Train set: {X_train.shape}")
    print(f"Validation set: {X_val.shape}")
    print(f"Test set: {X_test.shape}")
    
    # 4. Scale data
    print("Scaling data...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    
    # 5. Initialize tuner
    tuner = ComprehensiveHyperparameterTuner(X_train_scaled, y_train, X_val_scaled, y_val)
    
    # 6. Run comprehensive search
    print("Starting comprehensive random search...")
    best_model, best_params = tuner.comprehensive_search(max_trials=100)
    
    # 7. Run focused grid search
    print("\nStarting focused grid search...")
    best_model, best_params = tuner.grid_search_focused()
    
    # 8. Final evaluation on test set
    print(f"\n{'='*60}")
    print("FINAL EVALUATION")
    print(f"{'='*60}")
    print(f"Best parameters: {json.dumps(convert_numpy_types(best_params), indent=2)}")
    
    test_loss, test_mae = best_model.evaluate(X_test_scaled, y_test, verbose=0)
    print(f"Test Loss (MSE): {test_loss:.6f}")
    print(f"Test MAE: {test_mae:.6f}")
    
    # 9. Save everything
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_filename = f'best_comprehensive_model_{timestamp}.h5'
    scaler_filename = f'scaler_{timestamp}.pkl'
    
    best_model.save(model_filename)
    print(f"Best model saved as '{model_filename}'")
    
    import joblib
    joblib.dump(scaler, scaler_filename)
    print(f"Scaler saved as '{scaler_filename}'")
    
    # Save tuning results
    tuner.save_results()
    
    return best_model, best_params, tuner

if __name__ == "__main__":
    best_model, best_params, tuner = run_comprehensive_tuning()

