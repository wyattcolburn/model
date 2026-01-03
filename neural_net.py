import yaml
from datetime import datetime
import os
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
import tf2onnx

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import load_model
import glob
# import training_complete
from keras.callbacks import EarlyStopping
from keras.optimizers import Adam
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import StandardScaler

from datetime import datetime
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # force CPU for now

"""
The idea is that you give the program you dataset (which contains the csv files)
and it will create one large csv, but also create data for each ros bag

Input:
Directory of ros bag, or a list of ros bags. If all the bags are in one directory should be able to just give one input
"""


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


def train_and_save_model_combined(input_bag, model_path='combined.keras'):
    """This function is for taking large data set which has already formatted
    labels: cmd_v, cmd_w
    features: odom_odom_v, odom_odom_w, lidar0-1079, goal_local_goals_x	goal_local_goals_y	goal_local_goals_yaw

    """
    if len(input_bag) == 1:
        print("input bag just 1 length")
        input_bag = input_bag[0]

    try:
        features = pd.read_csv(f"{input_bag}/combined_features.csv", header=0)
        labels = pd.read_csv(f"{input_bag}/combined_labels.csv", header=0)
        print(f"shape of features : {features.shape} labels: {labels.shape}")

        # Verify they have matching number of rows
        if features.shape[0] != labels.shape[0]:
            print(
                f"WARNING: Mismatch - {features.shape[0]} features vs {labels.shape[0]} labels")
        else:
            print("Features and labels have matching row counts")

        X_train, X_val, y_train, y_val = train_test_split(
            features, labels, test_size=0.2, random_state=42)

        # Normalize data
        scaler = MinMaxScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)
        epochsVal = 50

        # early stopping
        early_stopping = EarlyStopping(
            monitor='val_loss',     # Change to a metric that exists in your model
            mode='min',           # Lower MAE is better, so use 'min'
            patience=5,
            min_delta=0.001,
            restore_best_weights=True
        )
        # Create and train model
        model = create_model(X_train_scaled.shape[1])
        history = model.fit(
            X_train_scaled, y_train,
            epochs=epochsVal,
            batch_size=256,
            callbacks=[],
            validation_data=(X_val_scaled, y_val)
        )

        # Save the model and scaler
        model.save(model_path)

        # Optional: Save scaler for inference

        # Plot training history
        graphs(history, "graph_test.png")
        # Save the scaler to use for inference
        np.save('combine_scaler_min.npy', scaler.min_)
        np.save('combine_scaler_scale_.npy', scaler.scale_)
        np.savetxt('combine_scaler_mins.txt', scaler.data_min_)
        np.savetxt('combine_scaler_max.txt', scaler.data_max_)
        # Save the combined data
        return scaler

    except FileNotFoundError as e:
        print(f"Error: Combined data files not found in {input_bag}")
        print("Make sure to run with --combine flag first to generate the combined data")
        exit(1)
    except pd.errors.EmptyDataError:
        print("Error: Combined data files are empty")
        exit(1)

    return


def train_and_save_model(input_bag, model_path='robot_model.keras'):
    """
    Train the model and save it to a file, this only takes one input bag (does not take combined data)

    Args:
        input_bag (str): Path to input data directory
        model_path (str): Path to save the trained model
    """
    # Load data
    training_lidar = pd.read_csv(
        f"{input_bag}/input_data/lidar_data.csv")  # no heaer (map frame)
    # odom_curren_v, odom_current_w (odom frame)
    training_odom = pd.read_csv(f"{input_bag}/input_data/odom_data.csv")
    # local_goal_x, local_goal_y, local_goal_yaw (map frame)
    training_local_goals = pd.read_csv(
        f"{input_bag}/input_data/local_goals.csv")
    training_labels = pd.read_csv(f"{input_bag}/input_data/cmd_vel_output.csv")

    # Preprocess data
    # training_lidar = training_odom.iloc[:-1, :]

    training_lidar = training_lidar[:-1]  # Remove last row
    training_odom = training_odom.iloc[:, [5, 6]]
    training_labels = training_labels.iloc[:, [2, 3]]
    training_local_goals = training_local_goals.iloc[:, [1, 2, 3]]

    print(
        f"shape of lidar, odom, labels, local_goals {training_lidar.shape} {training_odom.shape} {training_labels.shape} {training_local_goals.shape}")

    # Combine features
    features = pd.concat(
        [training_odom, training_local_goals, training_lidar], axis=1)

    # Split data
    X_train, X_val, y_train, y_val = train_test_split(
        features, training_labels, test_size=0.2, random_state=42)

    # Normalize data
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    epochsVal = 500

    # early stopping
    early_stopping = EarlyStopping(
        monitor='val_loss',     # Change to a metric that exists in your model
        mode='min',           # Lower MAE is better, so use 'min'
        patience=5,
        min_delta=0.001,
        restore_best_weights=True
    )
    # Create and train model
    model = create_model(X_train_scaled.shape[1])
    history = model.fit(
        X_train_scaled, y_train,
        epochs=epochsVal,
        batch_size=256,
        callbacks=[],
        validation_data=(X_val_scaled, y_val)
    )

    # Save the model and scaler
    model.save(model_path)

    # Optional: Save scaler for inference

    # Plot training history
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Training loss')
    plt.plot(history.history['val_loss'], label='Validation loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history['mae'], label='Training MAE')
    plt.plot(history.history['val_mae'], label='Validation MAE')
    plt.title('Training and Validation MAE')
    plt.xlabel('Epochs')
    plt.ylabel('Mean Absolute Error')
    plt.legend()

    plt.savefig(f"{input_bag}/MAE_{epochsVal}.png")
    plt.tight_layout()
    plt.show()

    # Save the scaler to use for inference
    np.save('scaler_min.npy', scaler.min_)
    np.save('scaler_scale_.npy', scaler.scale_)
    np.savetxt('scaler_mins.txt', scaler.data_min_)
    np.savetxt('scaler_max.txt', scaler.data_max_)
    # Save the combined data
    return scaler


def large_dataset(input_directory, single_dkr_flag, adaptive_flag):
    """
    Function to create large data sets
    Args:
        input_directory (list) of the directories that have a bag
            It is possible that data has already been created within a dkr has already been combined

    If there exists combined_features in both csv, then just add them
    Else:
        create the combined data sets first

    Returns:
        new directory with timestamp_combined_features, timestamp_combined_labels,
        model call timestamp.keras
        meta data, what this model was trainined on
    """

    print(input_directory)
    if single_dkr_flag:
        if isinstance(input_directory, str):
            print("directory is a str")
            if os.path.isdir(input_directory):
                subdirs = [f.path for f in os.scandir(
                    input_directory) if f.is_dir()]
            else:
                raise ValueError(f"Directory not found: {input_directory}")
        elif isinstance(input_directory, list):
            if len(input_directory) == 1:
                input_directory = input_directory[0]
                subdirs = [os.path.join(input_directory, d) for d in os.listdir(input_directory)
                           if os.path.isdir(os.path.join(input_directory, d))]
        else:
            raise ValueError(
                "input_directory must be a string (path) or list of paths")

    else:
        # Multiple directories mode - use the list directly
        if isinstance(input_directory, list):
            print("Multiple directories mode: using provided list")
            subdirs = input_directory
        elif isinstance(input_directory, str):
            print("Multiple directories mode: converting single string to list")
            subdirs = [input_directory]
        else:
            raise ValueError(
                "input_directory must be a string (path) or list of paths")
        print(f"Processing {len(subdirs)} directories")
    timestamp = datetime.now().strftime("%Y_%m_%d_%H_%M")
    new_dir = os.path.join("data_set", timestamp)
    os.makedirs(new_dir, exist_ok=True)

    yaml_data = {
        "run": {
            "timestamp": timestamp,
            "output_dir": new_dir,
        },
        "datasets": [],   # list of per-dataset records
        "combined": {},   # summary filled later
    }

    combined_features = None
    combined_labels = None
    total_rows = 0

    for data_dir in subdirs:
        feats_p = os.path.join(data_dir, "combined_features.csv")
        labs_p = os.path.join(data_dir, "combined_labels.csv")

        if os.path.exists(feats_p) and os.path.exists(labs_p):
            print(f"Files already exist in {data_dir}")

            # Load this dataset's CSVs
            df_feats = pd.read_csv(feats_p, header=0)
            df_labs = pd.read_csv(labs_p, header=0)

            # Append to combined
            if combined_features is None:
                combined_features = df_feats
                combined_labels = df_labs
            else:
                assert df_feats.shape[1] == combined_features.shape[1], \
                    f"Feature dimension mismatch: {df_feats.shape[1]} vs {combined_features.shape[1]}"
                combined_features = pd.concat(
                    [combined_features, df_feats], axis=0, ignore_index=True)
                combined_labels = pd.concat(
                    [combined_labels, df_labs], axis=0, ignore_index=True)

            total_rows += len(df_feats)

            # Record per-dataset metadata
            yaml_data["datasets"].append({
                "name": os.path.basename(os.path.normpath(data_dir)),
                "path": os.path.abspath(data_dir),
                "features_csv": os.path.abspath(feats_p),
                "labels_csv": os.path.abspath(labs_p),
                "features_shape": {"rows": int(df_feats.shape[0]), "cols": int(df_feats.shape[1])},
                "labels_shape": {"rows": int(df_labs.shape[0]), "cols": int(df_labs.shape[1])},
            })

            print(
                f"Combined features {combined_features.shape} and labels {combined_labels.shape}")

        else:
            print(
                f"Combined features do not exist within {data_dir}, will create")

            # Get seg_* directories
            try:
                seg_dirs = [d for d in os.listdir(data_dir)
                            if os.path.isdir(os.path.join(data_dir, d))]
                print("Found subdirectories:", seg_dirs)

                # Filter for seg_* directories only
                seg_dirs = [d for d in seg_dirs if d.startswith("seg_")]
                print("Filtered subdirectories:", seg_dirs)

                if not seg_dirs:
                    print(
                        f"No seg_* directories found in {data_dir}, skipping")
                    continue

            except OSError as e:
                print(f"Error accessing directory {data_dir}: {e}")
                continue

            # Process seg_* directories
            local_features = None
            local_labels = None

            for seg_dir in seg_dirs:
                try:
                    # Load data files
                    training_lidar = pd.read_csv(
                        f"{data_dir}/{seg_dir}/input_data/lidar_data.csv", header=0)
                    training_odom = pd.read_csv(
                        f"{data_dir}/{seg_dir}/input_data/odom_data.csv", header=0)
                    if adaptive_flag:
                        training_local_goals = pd.read_csv(f"{data_dir}/{seg_dir}/input_data/adaptive_local_goals.csv", header=0)
                    else:
                        training_local_goals = pd.read_csv(
                            f"{data_dir}/{seg_dir}/input_data/local_goals.csv", header=0)
                    
                    training_labels = pd.read_csv(
                        f"{data_dir}/{seg_dir}/input_data/cmd_vel_output.csv", header=0)

                    # Preprocess data
                    training_lidar = training_lidar.iloc[:-1, :]
                    training_odom = training_odom.iloc[:, [5, 6]]
                    if adaptive_flag:
                        training_local_goals = training_local_goals[:-1]
                    else: 
                        training_local_goals = training_local_goals.iloc[:, [
                            1, 2, 3]]
                    training_labels = training_labels.iloc[:, [2, 3]]

                    # Rename columns
                    training_odom.columns = [
                        f'odom_{col}' for col in training_odom.columns]
                    training_lidar.columns = [
                        f'lidar_{i}' for i in range(training_lidar.shape[1])]
                    training_local_goals.columns = [
                        f'goal_{col}' for col in training_local_goals.columns]

                    # Check minimum size
                    if training_lidar.shape[0] <= 200:
                        print(
                            f"{data_dir}/{seg_dir} is too small ({training_lidar.shape[0]} rows), skipping")
                        continue
                    print(f"shape of odom {training_odom.shape} lidar {training_lidar.shape} local goals {training_local_goals.shape}")

                    # Combine features
                    features = pd.concat(
                        [training_odom, training_local_goals, training_lidar], axis=1)
                    print(f"Features shape for {seg_dir}: {features.shape}")
                    print(
                        f"Labels shape for {seg_dir}: {training_labels.shape}")

                    # Add to local combined dataframes
                    if local_features is None:
                        local_features = features
                        local_labels = training_labels
                    else:
                        # Check dimensions match
                        if features.shape[1] != local_features.shape[1]:
                            print(
                                f"Feature dimension mismatch in {seg_dir}: {features.shape[1]} vs {local_features.shape[1]}, skipping")
                            continue

                        local_features = pd.concat(
                            [local_features, features], axis=0, ignore_index=True)
                        local_labels = pd.concat(
                            [local_labels, training_labels], axis=0, ignore_index=True)

                except Exception as e:
                    print(f"Error processing {seg_dir}: {e}")
                    continue

            # Save combined features for this directory and add to global dataset
            if local_features is not None and local_labels is not None:
                print(f"Creating combined features for {data_dir}")
                print(
                    f"Local dataset shape: features {local_features.shape}, labels {local_labels.shape}")

                # Save the combined files
                local_features.to_csv(
                    f'{data_dir}/combined_features.csv', mode='w', header=True, index=False)
                local_labels.to_csv(
                    f'{data_dir}/combined_labels.csv', mode='w', header=True, index=False)

                # Add to global combined dataset
                if combined_features is None:
                    combined_features = local_features
                    combined_labels = local_labels
                else:
                    # Check dimensions match
                    if local_features.shape[1] != combined_features.shape[1]:
                        print(
                            f"Feature dimension mismatch for {data_dir}: {local_features.shape[1]} vs {combined_features.shape[1]}")
                        print("Skipping this directory")
                        continue

                    combined_features = pd.concat(
                        [combined_features, local_features], axis=0, ignore_index=True)
                    combined_labels = pd.concat(
                        [combined_labels, local_labels], axis=0, ignore_index=True)

                total_rows += len(local_features)

                # Record metadata for this dataset
                yaml_data["datasets"].append({
                    "name": os.path.basename(os.path.normpath(data_dir)),
                    "path": os.path.abspath(data_dir),
                    "features_csv": os.path.abspath(f'{data_dir}/combined_features.csv'),
                    "labels_csv": os.path.abspath(f'{data_dir}/combined_labels.csv'),
                    "features_shape": {"rows": int(local_features.shape[0]), "cols": int(local_features.shape[1])},
                    "labels_shape": {"rows": int(local_labels.shape[0]), "cols": int(local_labels.shape[1])},
                    "seg_directories_processed": len(seg_dirs),
                })
            else:
                print(f"No valid data found in {data_dir}")

    # Fill combined summary
    if combined_features is not None:
        yaml_data["combined"] = {
            "features_shape": {"rows": int(combined_features.shape[0]), "cols": int(combined_features.shape[1])},
            "labels_shape": {"rows": int(combined_labels.shape[0]), "cols": int(combined_labels.shape[1])},
            "num_datasets": len(yaml_data["datasets"]),
            "total_rows": int(total_rows),
        }

        print(
            f"Final combined dataset: features {combined_features.shape}, labels {combined_labels.shape}")
    else:
        print("No valid data found across all directories")
        return

    # Write metadata YAML next to the run outputs
    meta_path = os.path.join(new_dir, "metadata.yaml")
    with open(meta_path, "w") as f:
        yaml.safe_dump(yaml_data, f, sort_keys=False)
    print("Wrote metadata:", meta_path)
    # Write metadata YAML next to the run outputs
    meta_path = os.path.join(new_dir, "metadata.yaml")
    with open(meta_path, "w") as f:
        yaml.safe_dump(yaml_data, f, sort_keys=False)

    print("Wrote metadata:", meta_path)

    X_train, X_val, y_train, y_val = train_test_split(
        combined_features, combined_labels, test_size=0.2, random_state=42)
    # Data all collected now we have to scale it
    scaler = MinMaxScaler()

    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    np.savetxt(os.path.join(
        new_dir, f"{timestamp}_scaler_mins.txt"),  scaler.data_min_)
    np.savetxt(os.path.join(
        new_dir, f"{timestamp}_scaler_maxs.txt"),  scaler.data_max_)

    print(f" combined features shape {combined_features.shape}")
    epochsVal = 150
    # early stopping
    # learning rate schedule

    early_stopping = EarlyStopping(
        monitor='val_loss',     # Change to a metric that exists in your model
        mode='min',           # Lower MAE is better, so use 'min'
        patience=5,
        min_delta=0.001,
        restore_best_weights=True
    )
    # Create and train model
    model = create_model(X_train_scaled.shape[1])
    history = model.fit(
        X_train_scaled, y_train,
        epochs=epochsVal,
        batch_size=256,
        callbacks=[],
        validation_data=(X_val_scaled, y_val)
    )
    model_path = f"{timestamp}.keras"
    # Save the model and scaler
    model.save(os.path.join(new_dir, model_path))
    convert_keras_onnx(os.path.join(new_dir, model_path),
                       os.path.join(new_dir, f"{timestamp}.onnx"))
    graphs(history, new_dir)
    # Optional: Save scaler for inference


def main():
    parser = argparse.ArgumentParser(
        description="Robot Model Training and Inference")
    parser.add_argument("input_bag", type=str, nargs='+',
                        help="Path to input data bag")
    parser.add_argument("--train", action="store_true", help="Train the model")
    parser.add_argument("--large", action="store_true",
                        help="Multiple data directories")
    parser.add_argument("--model", type=str,
                        help="Which model do you want to run")
    parser.add_argument("--combine", action="store_true",
                        help="add to big dataset")
    parser.add_argument("--train_combine", action="store_true",
                        help="train based on combined dkr")
    parser.add_argument("--single_dkr", action='store_true',
                        help="All training data within one directory")
    parser.add_argument("--adaptive", action='store_true',
                        help="All training data within one directory")
    args = parser.parse_args()

    if args.train_combine:
        train_and_save_model_combined(args.input_bag, args.model)
    if args.large:
        large_dataset(args.input_bag, args.single_dkr, args.adaptive)

    if args.train:
        # Train and save the model
        # if os.path.exists(f"{args.input_bag}/input_data"):
        #    print("training data already exists")
        # else:
        # training_complete.createFeatures(args.input_bag)
        train_combined(args.input_bag)
        # train_and_save_model(args.input_bag, args.model)


def graphs(history, filepath):

    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Training loss')
    plt.plot(history.history['val_loss'], label='Validation loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history['mae'], label='Training MAE')
    plt.plot(history.history['val_mae'], label='Validation MAE')
    plt.title('Training and Validation MAE')
    plt.xlabel('Epochs')
    plt.ylabel('Mean Absolute Error')
    plt.legend()

    plt.savefig(f"{filepath}/graphs.png")
    plt.tight_layout()
    plt.show()


def convert_keras_onnx(keras_model, output_model_name):
    filepath = keras_model
    m = load_model(filepath)

    flat_out = tf.nest.flatten(m.outputs)
    m.output_names = [t.name.split(":")[0] for t in flat_out]

    INPUT_DIM = m.input_shape[-1]
    spec = (tf.TensorSpec([None, INPUT_DIM], tf.float32, name="input"),)

    onnx_model, _ = tf2onnx.convert.from_keras(
        m, input_signature=spec, opset=17, output_path=output_model_name
    )
    print(f"conveted {output_model_name}")


def log_processing(input_directory, csv_file="proccessed_data.csv"):

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    if not os.path.exists(csv_file):
        pd.DataFrame(columns=['directory', 'timestamp']
                     ).to_csv(csv_file, index=False)

    new_entry = pd.DataFrame(
        [{'directory': input_directory, 'timestamp': timestamp}])
    new_entry.to_csv(csv_file, mode='a', header=False, index=False)


def read_processing(csv_file="proccessed_data.csv"):

    if not os.path.exists(csv_file):
        print("file not created yet")

    df = pd.read_csv(csv_file)
    return set(df['directory'].tolist())


if __name__ == "__main__":
    main()
