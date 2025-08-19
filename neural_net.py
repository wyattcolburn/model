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
from tensorflow import keras
from tensorflow.keras import layers
import glob
# import training_complete
from keras.callbacks import EarlyStopping
from keras.optimizers import Adam
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import StandardScaler

from datetime import datetime
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # force CPU for now


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


def train_combined(input_bag):
    """
    Train the model and save it to a file

    Args:
        input_bag (str): Path to input data directory
        model_path (str): Path to save the trained model
    """
    # Load data
    # no heaer (map frame)
    features = pd.read_csv(f"{input_bag}/combined_features.csv")
    training_labels = pd.read_csv(
        f"{input_bag}/combined_labels.csv")  # no heaer (map frame)

    print(
        f"feature size {features.shape} and label shape {training_labels.shape}")

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
    model.save("combine_08_14.keras")

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

    plt.savefig(f"combined_MAE.png")
    plt.tight_layout()
    plt.show()

    # Save the scaler to use for inference
    np.savetxt('combine_08_14_scaler_mins.txt', scaler.data_min_)
    np.savetxt('combine_08_14_scaler_max.txt', scaler.data_max_)
    # Save the combined data
    return scaler


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


def large_dataset(input_directory, model_path='robot_model.keras'):
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
    timestamp = datetime.now().strftime("%Y_%m_%d_%H_%M")
    new_dir = os.path.join("data_set", timestamp)
    os.makedirs(new_dir, exist_ok=True)

    # Resolve subdirs list
    if isinstance(input_directory, str):
        if os.path.isdir(input_directory):
            subdirs = [f.path for f in os.scandir(
                input_directory) if f.is_dir()]
        else:
            raise ValueError(f"Directory not found: {input_directory}")
    elif isinstance(input_directory, list):
        subdirs = input_directory
    else:
        raise ValueError(
            "input_directory must be a string (path) or list of paths")

    print(f"Processing {len(subdirs)} directories")

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
            print(f"value already exist in {data_dir}")

            # Load this dataset’s CSVs
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
                    [combined_labels,   df_labs],  axis=0, ignore_index=True)

            total_rows += len(df_feats)

            # Record per-dataset metadata
            yaml_data["datasets"].append({
                "name": os.path.basename(os.path.normpath(data_dir)),
                "path": os.path.abspath(data_dir),
                "features_csv": os.path.abspath(feats_p),
                "labels_csv":   os.path.abspath(labs_p),
                "features_shape": {"rows": int(df_feats.shape[0]), "cols": int(df_feats.shape[1])},
                "labels_shape":   {"rows": int(df_labs.shape[0]),  "cols": int(df_labs.shape[1])},
                # optional things you may want:
                # "feature_columns": list(df_feats.columns),
            })

            print(
                f"combined features {combined_features.shape} and labels {combined_labels.shape}")
        else:
            print(
                f"combined features does not exists within {data_dir} will create")
            # data dir will have seg_* which need to combined for combine_features
            seg_dirs = [d for d in os.listdir(data_dir)
                        if os.path.isdir(os.path.join(data_dir, d))]

            print("Found subdirectories:", seg_dirs)

            # If you only care about seg_* dirs
            seg_dirs = [d for d in seg_dirs if d.startswith("seg_")]

            print("Filtered subdirectories:", seg_dirs)
            # just for this directory so combined_features can be created within directory
            local_features = None
            local_labels = None
            for dir in seg_dirs:

                training_lidar = pd.read_csv(
                    f"{data_dir}/{dir}/input_data/lidar_data.csv", header=0)
                training_odom = pd.read_csv(
                    f"{data_dir}/{dir}/input_data/odom_data.csv", header=0)
                training_local_goals = pd.read_csv(
                    f"{data_dir}/{dir}/input_data/local_goals.csv", header=0)
                training_labels = pd.read_csv(
                    f"{data_dir}/{dir}/input_data/cmd_vel_output.csv", header=0)

                # Preprocess data
                training_lidar = training_lidar.iloc[:-1, :]
                training_odom = training_odom.iloc[:, [5, 6]]
                training_local_goals = training_local_goals.iloc[:, [1, 2, 3]]
                training_labels = training_labels.iloc[:, [2, 3]]
                training_odom.columns = [
                    f'odom_{col}' for col in training_odom.columns]
                training_lidar.columns = [
                    f'lidar_{i}' for i in range(training_lidar.shape[1])]
                training_local_goals.columns = [
                    f'goal_{col}' for col in training_local_goals.columns]

                # Combine features
                features = pd.concat(
                    [training_odom, training_local_goals, training_lidar], axis=1)
                print(f"feautres shape of {data_dir} : {features.shape}")
                print(f"labels shape of {data_dir} : {training_labels.shape}")

                # Add to combined dataframes
                if local_features is None:
                    local_features = features
                    local_labels = training_labels
                else:
                    # Make sure columns align before concatenating
                    assert features.shape[1] == local_features.shape[
                        1], f"Feature dimension mismatch: {features.shape[1]} vs {local_features.shape[1]}"
                    # Append rows
                    local_features = pd.concat(
                        [local_features, features], axis=0, ignore_index=True)
                    local_labels = pd.concat(
                        [local_labels, training_labels], axis=0, ignore_index=True)
            print(f"creating combined features for {data_dir}")
            features_df = pd.DataFrame(local_features)
            features_df.to_csv(
                f'{data_dir}/combined_features.csv', mode='w', header=True, index=False)
            labels_df = pd.DataFrame(local_labels)
            labels_df.to_csv(
                f'{data_dir}/combined_labels.csv', mode='w', header=True, index=False)

            if combined_features is None:
                combined_features = local_features
                combined_labels = local_labels

            else:

                assert df_feats.shape[1] == combined_features.shape[1], \
                    f"Feature dimension mismatch: {df_feats.shape[1]} vs {combined_features.shape[1]}"
                combined_features = pd.concat(
                    [combined_features, df_feats], axis=0, ignore_index=True)
                combined_labels = pd.concat(
                    [combined_labels,   df_labs],  axis=0, ignore_index=True)

            total_rows += len(df_feats)
            yaml_data["datasets"].append({
                "name": os.path.basename(os.path.normpath(data_dir)),
                "path": os.path.abspath(data_dir),
                "features_csv": os.path.abspath(feats_p),
                "labels_csv":   os.path.abspath(labs_p),
                "features_shape": {"rows": int(df_feats.shape[0]), "cols": int(df_feats.shape[1])},
                "labels_shape":   {"rows": int(df_labs.shape[0]),  "cols": int(df_labs.shape[1])},
                # optional things you may want:
                # "feature_columns": list(df_feats.columns),
            })

    # Fill combined summary
    if combined_features is not None:
        yaml_data["combined"] = {
            "features_shape": {"rows": int(combined_features.shape[0]), "cols": int(combined_features.shape[1])},
            "labels_shape":   {"rows": int(combined_labels.shape[0]),  "cols": int(combined_labels.shape[1])},
            "num_datasets": len(yaml_data["datasets"]),
            "total_rows": int(total_rows),
        }

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
    epochsVal = 500
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
    graphs(history, new_dir)
    # Optional: Save scaler for inference
    """
    # Plot training history

    return combined_features, combined_labels

        training_lidar = pd.read_csv(
            f"{data_dir}/input_data/lidar_data.csv", header = 0)
        training_odom = pd.read_csv(
            f"{data_dir}/input_data/odom_data.csv", header = 0)
        training_local_goals = pd.read_csv(
            f"{data_dir}/input_data/local_goals.csv", header = 0)
        training_labels = pd.read_csv(
            f"{data_dir}/input_data/cmd_vel_output.csv", header = 0)

        # Preprocess data
        training_lidar = training_lidar.iloc[:-1,:]
        training_odom = training_odom.iloc[:, [5,6]]
        training_local_goals = training_local_goals.iloc[:, [1,2,3]]
        training_labels = training_labels.iloc[:, [2,3]]
        training_odom.columns = [
            f'odom_{col}' for col in training_odom.columns]
        training_lidar.columns = [
            f'lidar_{i}' for i in range(training_lidar.shape[1])]
        training_local_goals.columns = [
            f'goal_{col}' for col in training_local_goals.columns]

        # Combine features
        features = pd.concat(
            [training_odom, training_lidar, training_local_goals], axis=1)
        print(f"feautres shape of {data_dir} : {features.shape}")
        print(f"labels shape of {data_dir} : {training_labels.shape}")


        # Add to combined dataframes
        if combined_features is None:
            combined_features = features
            combined_labels = training_labels
        else:
            # Make sure columns align before concatenating
            assert features.shape[1] == combined_features.shape[
                1], f"Feature dimension mismatch: {features.shape[1]} vs {combined_features.shape[1]}"
            # Append rows
            combined_features = pd.concat(
                [combined_features, features], axis=0, ignore_index=True)
            combined_labels = pd.concat(
                [combined_labels, training_labels], axis=0, ignore_index=True)
    # Split the data into training and validation
    print(
        f"shape of combined feautres {combined_features.shape} and labels {combined_labels.shape}")
    """


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


def main():
    parser = argparse.ArgumentParser(
        description="Robot Model Training and Inference")
    parser.add_argument("input_bag", type=str, nargs='+',
                        help="Path to input data bag")
    parser.add_argument("--train", action="store_true", help="Train the model")
    parser.add_argument("--predict", action="store_true", help="Run inference")
    parser.add_argument("--large", action="store_true",
                        help="Multiple data directories")
    parser.add_argument("--model", type=str,
                        help="Which model do you want to run")
    parser.add_argument("--combine", action="store_true",
                        help="add to big dataset")
    parser.add_argument("--train_combine", action="store_true",
                        help="train based on combined dkr")
    args = parser.parse_args()

    if args.train_combine:
        train_and_save_model_combined(args.input_bag, args.model)
    if args.large:
        large_dataset(args.input_bag)

    if args.train:
        # Train and save the model
        # if os.path.exists(f"{args.input_bag}/input_data"):
        #    print("training data already exists")
        # else:
        # training_complete.createFeatures(args.input_bag)
        train_combined(args.input_bag)
        # train_and_save_model(args.input_bag, args.model)

    if args.predict:
        # Example of loading data for prediction
        # You'll need to prepare your input data similarly to training data
        # if os.path.exists(f"{args.input_bag}/input_data"):
       #     print("training data already exists")
        # else:

        training_lidar = pd.read_csv(
            f"{args.input_bag}/input_bag/lidar_data.csv")  # no heaer (map frame)
        # odom_curren_v, odom_current_w (odom frame)
        training_odom = pd.read_csv(
            f"{args.input_bag}/input_bag/odom_data.csv")
        # local_goal_x, local_goal_y, local_goal_yaw (map frame)
        training_local_goals = pd.read_csv(
            f"{args.input_bag}/input_bag/local_goals.csv")

        # Preprocess data
        # training_lidar = training_odom.iloc[:-1, :]

        training_lidar = training_lidar[:-1]  # Remove last row
        training_odom = training_odom.iloc[:, [5, 6]]
        training_local_goals = training_local_goals.iloc[:, [1, 2, 3]]

        print(
            f"shape of lidar, odom, labels, local_goals {training_lidar.shape} {training_odom.shape} {training_local_goals.shape}")

        # Combine features
        features = pd.concat(
            [training_odom, training_local_goals, training_lidar], axis=1)

        # Make predictions
        predictions = load_model_and_predict(features)
        print("Predictions:", predictions)
        output_dir = os.path.join(args.input_bag, "output_data")
        os.makedirs(output_dir, exist_ok=True)

        np.savetxt(os.path.join(output_dir, "cmd_vel.csv"),
                   predictions, delimiter=",")
        print("written output cmd values")

    if args.combine:

        input_directory = args.input_bag

        if isinstance(input_directory, str):
            # If it's a string, assume it's a parent directory and get subdirectories
            if os.path.isdir(input_directory):
                subdirs = [f.path for f in os.scandir(
                    input_directory) if f.is_dir()]
            else:
                raise ValueError(f"Directory not found: {input_directory}")
        elif isinstance(input_directory, list):
            # If it's a list, use it directly
            subdirs = input_directory
        else:
            raise ValueError(
                "input_directory must be a string (path) or list of paths")

        # print(f"Processing {len(subdirs)} directories")
        # subdirs = glob.glob(os.path.join(
        #     input_directory, "**/"), recursive=True)
        subdirs = [d for d in subdirs if os.path.isdir(d)]
        for data_dir in subdirs:
            print(f"printing dkr {data_dir}")
        combined_features = None
        combined_labels = None
        for data_dir in subdirs:
            # values already been calculated
            if os.path.exists(f"{data_dir}/input_data"):
                print(f"value already exist in {data_dir}")
            else:
                print(f"training data does not exist")

            training_lidar = pd.read_csv(
                f"{data_dir}/input_data/lidar_data.csv", header=0)
            training_odom = pd.read_csv(
                f"{data_dir}/input_data/odom_data.csv", header=0)
            training_local_goals = pd.read_csv(
                f"{data_dir}/input_data/local_goals.csv", header=0)
            training_labels = pd.read_csv(
                f"{data_dir}/input_data/cmd_vel_output.csv", header=0)

            # Preprocess data
            training_lidar = training_lidar.iloc[:-1, :]
            training_odom = training_odom.iloc[:, [5, 6]]
            training_local_goals = training_local_goals.iloc[:, [1, 2, 3]]
            training_labels = training_labels.iloc[:, [2, 3]]
            training_odom.columns = [
                f'odom_{col}' for col in training_odom.columns]
            training_lidar.columns = [
                f'lidar_{i}' for i in range(training_lidar.shape[1])]
            training_local_goals.columns = [
                f'goal_{col}' for col in training_local_goals.columns]

            # Combine features
            features = pd.concat(
                [training_odom, training_local_goals, training_lidar], axis=1)
            print(f"feautres shape of {data_dir} : {features.shape}")
            print(f"labels shape of {data_dir} : {training_labels.shape}")

            # Add to combined dataframes
            if combined_features is None:
                combined_features = features
                combined_labels = training_labels
            else:
                # Make sure columns align before concatenating
                assert features.shape[1] == combined_features.shape[
                    1], f"Feature dimension mismatch: {features.shape[1]} vs {combined_features.shape[1]}"
                # Append rows
                combined_features = pd.concat(
                    [combined_features, features], axis=0, ignore_index=True)
                combined_labels = pd.concat(
                    [combined_labels, training_labels], axis=0, ignore_index=True)
        # Make sure same size
        print(
            f"shape of combined feautres {combined_features.shape} and labels {combined_labels.shape}")

        if os.path.exists(f"{args.input_bag}/combined_features.csv"):

            print("file already exists")
            # check if this dkr has already been added
            dkr_list = read_processing()
            if input_directory in dkr_list:
                print("already added this data to combined dataset")
                return

            print(f"adding data from {input_directory}")
            # do not need to create a header
            features_df = pd.DataFrame(combined_features)
            features_df.to_csv('combined_features.csv',
                               mode='a', header=False, index=False)

            labels_df = pd.DataFrame(combined_labels)
            labels_df.to_csv('combined_labels.csv', mode='a',
                             header=False, index=False)
            log_processing(input_directory)

        else:
            print("creating a file for features")
            features_df = pd.DataFrame(combined_features)
            features_df.to_csv(
                f'{args.input_bag}/combined_features.csv', mode='w', header=True, index=False)

            labels_df = pd.DataFrame(combined_labels)
            labels_df.to_csv(
                f'{args.input_bag}/combined_labels.csv', mode='w', header=True, index=False)
            print("creating log file")
            log_processing(input_directory)
            dkr_list = read_processing()
            for dkr in dkr_list:
                print(dkr)
        print("success")


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
