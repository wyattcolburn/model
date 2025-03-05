import os
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

import training_complete
def create_model(input_shape):
    """
    Create the neural network model architecture
    
    Args:
        input_shape (int): Number of input features
    
    Returns:
        keras.Sequential: Compiled model
    """
    model = keras.Sequential([
        layers.Input(shape=(input_shape,)),
        layers.Dense(256, activation='relu'),
        layers.Dense(256, activation='relu'),
        layers.Dense(256, activation='relu'),
        layers.Dense(2)  # Output layer with 2 neurons
    ])
    
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model

def train_and_save_model(input_bag, model_path='robot_model.keras'):
    """
    Train the model and save it to a file
    
    Args:
        input_bag (str): Path to input data directory
        model_path (str): Path to save the trained model
    """
    # Load data
    training_lidar = pd.read_csv(f"{input_bag}/input_data/lidar_data.csv")
    training_odom = pd.read_csv(f"{input_bag}/input_data/odom_data.csv")
    training_local_goals = pd.read_csv(f"{input_bag}/input_data/local_goals.csv")
    training_labels = pd.read_csv(f"{input_bag}/_data/input_cmd_vel_output.csv")
   
    # Preprocess data
    training_lidar = training_lidar.iloc[:,1:]
    training_odom = training_odom.iloc[:, [1,2,4,5]]
    training_labels = training_labels.iloc[:, [1,2]]
    training_odom = training_odom.iloc[:-1,:]
    training_labels = training_labels.iloc[:-1, :]
    training_local_goals = training_local_goals.iloc[:-1, :]
    
    # Combine features
    features = pd.concat([training_odom, training_lidar, training_local_goals], axis=1)
    
    # Split data
    X_train, X_val, y_train, y_val = train_test_split(features, training_labels, test_size=0.2, random_state=42)
    
    # Normalize data
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    
    # Create and train model
    model = create_model(X_train_scaled.shape[1])
    history = model.fit(
        X_train_scaled, y_train, 
        epochs=50, 
        batch_size=32, 
        validation_data=(X_val_scaled, y_val)
    )
    
    # Save the model and scaler
    model.save(model_path)
    
    # Optional: Save scaler for inference
    
    # Plot training history
    plt.figure(figsize=(12,4))
    plt.subplot(1,2,1)
    plt.plot(history.history['loss'], label='Training loss')
    plt.plot(history.history['val_loss'], label='Validation loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.subplot(1,2,2)
    plt.plot(history.history['mae'], label='Training MAE')
    plt.plot(history.history['val_mae'], label='Validation MAE')
    plt.title('Training and Validation MAE')
    plt.xlabel('Epochs')
    plt.ylabel('Mean Absolute Error')
    plt.legend()
    
    plt.savefig(f"{input_bag}/_data/MAE.png")
    plt.tight_layout()
    plt.show()
    
    np.save('scaler_min.npy', scaler.min_)
    np.save('scaler_scale_.npy', scaler.scale_)
    return scaler

def load_model_and_predict(input_data, model_path='robot_model.keras', scaler_path='feature_scaler.joblib'):
    """
    Load trained model and make predictions
    
    Args:
        input_data (pd.DataFrame): Input features for prediction
        model_path (str): Path to saved model
        scaler_path (str): Path to saved scaler
    
    Returns:
        np.ndarray: Predicted values
    """
    # Load the saved model
    model = keras.models.load_model(model_path)
    

    # Load the scaler
    scaler = MinMaxScaler()
    scaler.min_ = np.load('scaler_min.npy')
    scaler.scale_ = np.load('scaler_scale_.npy')
    
   
    # Scale the input data
    input_scaled = scaler.transform(input_data)
    
    # Make predictions
    predictions = model.predict(input_scaled)
    print(f"shape of predictions {predictions.shape}") 
    return predictions

def main():
    parser = argparse.ArgumentParser(description="Robot Model Training and Inference")
    parser.add_argument("input_bag", type=str, help="Path to input data bag")
    parser.add_argument("--train", action="store_true", help="Train the model")
    parser.add_argument("--predict", action="store_true", help="Run inference")
    
    args = parser.parse_args()
    
    if args.train:
        # Train and save the model
        if os.path.exists(f"{args.input_bag}/input_data"):
            print("training data already exists")
        else: 
            training_complete.createFeatures(args.input_bag)
        train_and_save_model(args.input_bag)
    
    if args.predict:
        # Example of loading data for prediction
        # You'll need to prepare your input data similarly to training data
        if os.path.exists(f"{args.input_bag}/input_data"):
            print("training data already exists")
        else: 
            training_complete.createFeatures(args.input_bag)
        
        training_lidar = pd.read_csv(f"{args.input_bag}/input_data/lidar_data.csv")
        training_odom = pd.read_csv(f"{args.input_bag}/input_data/odom_data.csv")
        training_local_goals = pd.read_csv(f"{args.input_bag}/input_data/local_goals.csv")
        
        # Preprocess data (similar to training preprocessing)
        training_lidar = training_lidar.iloc[:,1:]
        training_odom = training_odom.iloc[:, [1,2,4,5]]
        training_local_goals = training_local_goals.iloc[:-1, :]
        
        print(f"lidar shape {training_lidar.shape} odom shape {training_odom.shape}   local_goals shape {training_local_goals.shape}")
        # Combine features (ensure this matches training feature combination)
        features = pd.concat([training_odom, training_lidar, training_local_goals], axis=1)
        
        # Make predictions
        predictions = load_model_and_predict(features)
        print("Predictions:", predictions)
        output_dir = os.path.join(args.input_bag, "output_data")
        os.makedirs(output_dir, exist_ok = True)

        np.savetxt(os.path.join(output_dir, "cmd_vel.csv"), predictions, delimiter=",")
        print("written output cmd values")

if __name__ == "__main__":
    main()
