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
import glob
import training_complete
from keras.callbacks import EarlyStopping
from keras.optimizers import Adam
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
    
    lr_schedule = keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=1e-3,
    decay_steps=10000,
    decay_rate=0.9)

    ourAdam = keras.optimizers.Adam(
    learning_rate=lr_schedule,
    name="adam")
    print("*****************************OUR ADAM WITH SCHEDULING")
    model.compile(optimizer=ourAdam, loss='mse', metrics=['mae'])
    return model
def modulation(w1, w2, p):

    """
    Requires odom data, and obstacle data, then with current_v, current_w need to create
    a safety function for modulation

    """
    pass
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
    training_labels = pd.read_csv(f"{input_bag}/input_data/cmd_vel_output.csv")
   
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
    epochsVal = 500

    # early stopping
    early_stopping = EarlyStopping(monitor='val_accuracy', patience=5, min_delta=0.001, restore_best_weights=True)
    # Create and train model
    model = create_model(X_train_scaled.shape[1])
    history = model.fit(
        X_train_scaled, y_train, 
        epochs=epochsVal, 
        batch_size=32, 
        callbacks=[callback],
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
    
    plt.savefig(f"{input_bag}/_data/MAE_{epochsVal}.png")
    plt.tight_layout()
    plt.show()
   
    # Save the scaler to use for inference
    np.save('scaler_min.npy', scaler.min_)
    np.save('scaler_scale_.npy', scaler.scale_)
    
    # Save the combined data
    return scaler



def combined_training_data(input_directory, model_path='robot_model.keras'):
    """
    Want to combined training data from multiple directories within training directory

    Args:
        input_directory (list) of the directories that have a bag

    Returns:
        combined_features, combined_labels,

    Saves:
        Also saves this data in combined_features dkr
    """

    
    combined_features = None
    combined_labels = None
    for data_dir in input_directory:
        if os.path.exists(f"{data_dir}/input_data"): # values already been calculated
            print(f"value already exist in {data_dir}")
        else:
            training_complete.createFeatures(data_dir)
            print(f"adding training data {data_dir}")

        training_lidar = pd.read_csv(f"{data_dir}/input_data/lidar_data.csv")
        training_odom = pd.read_csv(f"{data_dir}/input_data/odom_data.csv")
        training_local_goals = pd.read_csv(f"{data_dir}/input_data/local_goals.csv")
        training_labels = pd.read_csv(f"{data_dir}/input_data/cmd_vel_output.csv")
   
        # Preprocess data
        training_lidar = training_lidar.iloc[:,1:]
        training_odom = training_odom.iloc[:, [1,2,4,5]]
        training_labels = training_labels.iloc[:, [1,2]]
        training_odom = training_odom.iloc[:-1,:]
        training_labels = training_labels.iloc[:-1, :]
        training_local_goals = training_local_goals.iloc[:-1, :]
        
        # Combine features
        features = pd.concat([training_odom, training_lidar, training_local_goals], axis=1)
        feature_columns = [f'feature_{i}' for i in range(features.shape[1])]
        features.columns = feature_columns
        print(f"feautres shape of {data_dir} : {features.shape}")     
    

        # Add to combined dataframes
        if combined_features is None:
            combined_features = features
            combined_labels = training_labels
        else:
            # Append rows
            combined_features = pd.concat([combined_features, features], axis=0, ignore_index=True)
            combined_labels = pd.concat([combined_labels, training_labels], axis=0, ignore_index=True)
    # Split the data into training and validation
     
    X_train, X_val, y_train, y_val = train_test_split(combined_features, combined_labels, test_size=0.2, random_state=42)
    #Data all collected now we have to scale it
    scaler = MinMaxScaler()
    
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    np.save('scaler_min.npy', scaler.min_)
    np.save('scaler_scale_.npy', scaler.scale_)
    combined_dir = "combined_dir"
    os.makedirs(combined_dir, exist_ok=True)
    combined_features.to_csv(f"{combined_dir}/combined_features.csv", index=False)
    combined_labels.to_csv(f"{combined_dir}/combined_labels.csv", index=False)
    print(f"Saved combined dataset to {combined_dir}")
    
    print(f" combined features shape {combined_features.shape}")
    epochsVal = 1000
    # early stopping
    early_stopping = EarlyStopping(monitor='val_accuracy', mode='min', patience=5, restore_best_weights=True)
    #learning rate schedule
    
    # Create and train model
    model = create_model(X_train_scaled.shape[1])
    history = model.fit(
        X_train_scaled, y_train, 
        epochs=epochsVal, 
        batch_size=64, 
        callbacks=[early_stopping],
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
    
    plt.savefig(f"{data_dir}/MAE_{epochsVal}.png")
    plt.tight_layout()
    plt.show()
    
    return combined_features, combined_labels

        # Now we have the data

def load_model_and_predict(input_data, model_path='robot_model_adv.keras', scaler_path='feature_scaler.joblib'):
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
     

def input_directory_source():

    """ Want to return a list of directories which are within the training folder
    which are ros bags"""

    os.makedirs("training", exist_ok = True)

    dirs = [d for d in glob.glob(f"training_temp2/*/") if os.path.isdir(d)] 

    print(dirs)

    return dirs
def main():
    parser = argparse.ArgumentParser(description="Robot Model Training and Inference")
    parser.add_argument("input_bag", type=str, help="Path to input data bag")
    parser.add_argument("--train", action="store_true", help="Train the model")
    parser.add_argument("--predict", action="store_true", help="Run inference")
    parser.add_argument("--add", action="store_true", help="Add to training dataset") 
    parser.add_argument("model",type=str, help="Which model do you want to run")
    args = parser.parse_args()
    

    
    if args.add:
        dir = input_directory_source()
        combined_training_data(dir, args.model)

    if args.train:
        # Train and save the model
        #if os.path.exists(f"{args.input_bag}/input_data"):
        #    print("training data already exists")
        #else: 
        training_complete.createFeatures(args.input_bag)
        train_and_save_model(args.input_bag, args.model)
    
    if args.predict:
        # Example of loading data for prediction
        # You'll need to prepare your input data similarly to training data
        #if os.path.exists(f"{args.input_bag}/input_data"):
       #     print("training data already exists")
        #else: 
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
