import numpy as np
import pandas as pd
import mysql.connector
from mysql.connector import Error
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA

# MySQL database connection function
def connect_to_database():
    try:
        # Establishing connection to the database
        connection = mysql.connector.connect(
            host='crypto-matter.c5eq66ogk1mf.eu-central-1.rds.amazonaws.com',
            database='Crypto',
            user='Jing',  # Replace with your actual first name
            password='Crypto12!'
        )

        if connection.is_connected():
            db_info = connection.get_server_info()
            print("Connected to MySQL database, MySQL Server version: ", db_info)
            return connection

    except Error as e:
        print("Error while connecting to MySQL", e)
        return None

# Function to query merged data from crypto_lending_borrowing and crypto_price tables
def query_merged_crypto_data(connection):
    query = """
    SELECT clb.lending_rate, clb.borrowing_rate, clb.utilization_rate, clb.stable_borrow_rate,
    cp.*, usb.yield
    FROM crypto_lending_borrowing clb
    JOIN crypto_price cp 
        ON clb.crypto_symbol = cp.crypto_symbol
        AND clb.date = cp.date
    LEFT JOIN US_Bond_Yield usb
        ON clb.date = usb.date
    WHERE UPPER(clb.crypto_symbol) IN ('1INCHUSDT', 'BALUSDT', 'BATUSDT', 'CRVUSDT', 'ENJUSDT', 'ENSUSDT', 'KNCUSDT', 'LINKUSDT', 'MANAUSDT', 'MKRUSDT', 'RENUSDT', 'SNXUSDT', 'UNIUSDT', 'WBTCUSDT', 'YFIUSDT', 'ZRXUSDT')
    """
    cursor = connection.cursor()

    try:
        # Execute the query
        cursor.execute(query)

        # Fetch all results
        results = cursor.fetchall()

        # Get column names from cursor description
        columns = [desc[0] for desc in cursor.description]

        # Convert results to a Pandas DataFrame
        df = pd.DataFrame(results, columns=columns)

        return df

    except Error as e:
        print(f"Error: {e}")
        return None
    finally:
        cursor.close()

# Function to close the database connection
def query_quit(connection):
    if connection.is_connected():
        connection.close()
        print("MySQL connection is closed")

# Define a function to calculate outlier bounds using IQR
def calculate_iqr_bounds(series, multiplier=1.5):
    Q1 = series.quantile(0.25)
    Q3 = series.quantile(0.75)
    IQR = Q3 - Q1
    if multiplier == 'remove':
        lower_bound = min(series) - 1
        upper_bound = max(series) + 1
    else:
        lower_bound = max(Q1 - multiplier * IQR, 0)
        upper_bound = Q3 + multiplier * IQR
    return lower_bound, upper_bound

# calculate returns on valid windows
def calculate_hourly_returns(df, date_col, close_col):
    """
    Calculates returns based on the close price, only if the date difference is 1 hour.

    Args:
        df (pd.DataFrame): The DataFrame containing the time series data.
        date_col (str): The name of the datetime column.
        close_col (str): The name of the close price column.

    Returns:
        pd.Series: A Series containing the calculated returns or None for invalid rows.
    """
    # Ensure the date column is in datetime format
    df[date_col] = pd.to_datetime(df[date_col])
    
    # Sort by date to ensure sequential order
    df = df.sort_values(by=date_col).reset_index(drop=True)
    
    # Calculate the time difference between consecutive rows in hours
    time_diff = df[date_col].diff().dt.total_seconds() / 3600
    
    # Calculate returns only for rows where time_diff == 1 hour
    returns = np.where(
        time_diff == 1,
        (df[close_col] - df[close_col].shift(1)) / df[close_col].shift(1),
        None
    )
    
    return pd.Series(returns, index=df.index)

# now we have a dataframe that does not have any NA and ay outlier, but its time series is corrupted, therefore we need valid windows
def extract_valid_windows(df, date_col, input_window, target_window, input_columns, target_columns):
    """
    Extracts valid windows from a time series DataFrame for LSTM training.
    
    Args:
        df (pd.DataFrame): The time series DataFrame with a datetime column.
        date_col (str): The name of the datetime column.
        input_window (int): The number of timesteps for the input sequence.
        target_window (int): The number of timesteps for the target sequence.
        input_columns (list of str): List of column names to include in the input data.
        target_columns (list of str): List of column names to include in the target data.
        
    Returns:
        inputs (list of np.ndarray): List of valid input sequences.
        targets (list of np.ndarray): List of corresponding target sequences.
    """
    # Sort by the datetime column to ensure the time series is ordered
    df = df.sort_values(by=date_col).reset_index(drop=True)
    
    # Ensure the datetime column is in pandas datetime format
    df[date_col] = pd.to_datetime(df[date_col])
    
    # Identify valid consecutive rows (1-hour apart)
    time_diffs = df[date_col].diff().dt.total_seconds()
    valid_indices = time_diffs == 3600  # 1 hour = 3600 seconds
    
    # Mark valid sequences
    valid_sequence_flags = valid_indices | valid_indices.shift(-1, fill_value=False)
    df = df[valid_sequence_flags].reset_index(drop=True)

    # Prepare inputs and targets
    inputs, targets = [], []
    total_window = input_window + target_window

    for i in range(len(df) - total_window + 1):
        # Extract a potential window of size `total_window`
        window = df.iloc[i:i+total_window]
        
        # Check if all rows in the window are 1-hour apart
        if (window[date_col].diff().dt.total_seconds()[1:] == 3600).all():
            # Split into input and target based on specified columns
            input_data = window.iloc[:input_window][input_columns].values
            target_data = window.iloc[input_window:][target_columns].values
            inputs.append(input_data)
            targets.append(target_data)

    return np.array(inputs), np.array(targets)


# embedding the name
def create_llm_embeddings(dataframe, col, n_components=10):
    model = SentenceTransformer('all-MiniLM-L6-v2')

    unique_values = dataframe[col].unique()
    
    # Get embeddings for the unique values
    embeddings = model.encode(unique_values, show_progress_bar=False)

    # Apply PCA to reduce dimensionality
    pca = PCA(n_components=n_components)
    reduced_embeddings = pca.fit_transform(embeddings)

    # Create a DataFrame to hold the reduced embeddings
    reduced_embeddings_df = pd.DataFrame(reduced_embeddings, columns=[f'{col}_embedding_{i+1}' for i in range(n_components)])

    reduced_embeddings_df[col] = unique_values

    dataframe = dataframe.merge(reduced_embeddings_df, on=col, how='left')

    return dataframe


# adding time attributes
def create_cyclical_encodings(df, date_col):
    
    days_in_month_dict = {
    1: 31, 2: 28, 3: 31, 4: 30, 5: 31, 6: 30,
    7: 31, 8: 31, 9: 30, 10: 31, 11: 30, 12: 31}

    df["Year"] = df[date_col].dt.year
    df["Month"] = df[date_col].dt.month
    df["Day"] = df[date_col].dt.day
    df["Hour"] = df[date_col].dt.hour
    df["DayofWeek"] = df[date_col].dt.dayofweek

    # Vectorized leap year handling and days in month calculation
    leap_year_mask = (df['Year'] % 4 == 0) & ((df['Year'] % 100 != 0) | (df['Year'] % 400 == 0))
    df['days_in_month'] = df['Month'].map(days_in_month_dict)
    
    # Adjust February for leap years
    df.loc[leap_year_mask & (df['Month'] == 2), 'days_in_month'] = 29

    df["Month_Sine"] = np.sin(2* np.pi * df["Month"] / 12)
    df["Month_Cosine"] = np.cos(2* np.pi * df["Month"] / 12)
    
    df["Day_Sine"] = np.sin(2* np.pi * df["Day"] / df['days_in_month'])
    df["Day_Cosine"] = np.cos(2* np.pi * df["Day"] / df['days_in_month'])
    
    df["Hour_Sine"] = np.sin(2* np.pi * df["Hour"] / 24)
    df["Hour_Cosine"] = np.cos(2* np.pi * df["Hour"] / 24)
    
    df["DayofWeek_Sine"] = np.sin(2* np.pi * df["DayofWeek"] / 7)
    df["DayofWeek_Cosine"] = np.cos(2* np.pi * df["DayofWeek"] / 7)

    df.drop(columns=["Month", "Day", "days_in_month", "Hour", "DayofWeek", "Year"], inplace=True)

    return df






import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout, Masking, Reshape, Layer, Lambda, Concatenate, LayerNormalization, MultiHeadAttention
from tensorflow.keras.models import Model
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, classification_report
from tensorflow.keras.optimizers import Adadelta
from keras.layers import BatchNormalization

def train_v1(inputs, outputs, test_size=0.2, valid_size=0.25, epochs=50, batch_size=100, d1=0.1, d2 = 0.05, cell_size = 80):
    # Clearing the TensorFlow session to ensure the model starts with fresh weights and biases
    tf.keras.backend.clear_session()
    n_classes = 3

    cell_size_1 = cell_size
    cell_size_2 = cell_size_1//2

    # Splitting the data into train+validation and test sets
    X_train_valid, X_test, y_train_valid, y_test = train_test_split(
        inputs, outputs, test_size=test_size)

    # Splitting the train+validation set into separate training and validation sets
    X_train, X_valid, y_train, y_valid = train_test_split(
        X_train_valid, y_train_valid, test_size=valid_size)

    # Model definition
    inputs = Input(shape=(X_train.shape[1], X_train.shape[2]))
    Lstm_layer_1 = LSTM(cell_size_1, return_sequences=True, stateful=False, use_cudnn=False)(inputs)
    Batch_norm_1 = BatchNormalization()(Lstm_layer_1)
    Dropout_layer_1 = Dropout(d1)(Batch_norm_1)
    Lstm_layer_2 = LSTM(cell_size_2, return_sequences=False, stateful=False, use_cudnn=False)(Dropout_layer_1)  # just halved
    Batch_norm_2 = BatchNormalization()(Lstm_layer_2)
    Drouput_layer_2 = Dropout(d2)(Batch_norm_2)
    predictions = Dense(y_train.shape[1]*y_train.shape[2]*n_classes, activation='softmax')(Drouput_layer_2)
    predictions_reshaped = Reshape((y_train.shape[1], y_train.shape[2], n_classes),name="class")(predictions)
    LSTM_base = Model(inputs=inputs, outputs=predictions_reshaped)


    # optimizer
    optimizer = Adadelta(
    learning_rate=1.0,
    rho=0.8,
    epsilon=1e-7)      # Default , to prevent division by zero)


    # Compiling the model
    def sparse_crossentropy_masked(y_true, y_pred):
        y_true = tf.cast(y_true, tf.int32)
        y_pred = tf.cast(y_pred, tf.float32)
        mask = tf.not_equal(y_true, -50)
        y_true_masked = tf.boolean_mask(y_true, mask)
        y_pred_masked = tf.boolean_mask(y_pred, mask)
        loss = tf.compat.v1.losses.sparse_softmax_cross_entropy(y_true_masked, y_pred_masked)
        return tf.reduce_mean(loss)
    
    LSTM_base.compile(
        optimizer=optimizer,
        loss=sparse_crossentropy_masked,
        metrics=['accuracy'])

    # Training the model
    history = LSTM_base.fit(x=X_train, y=y_train,
                    validation_data=(X_valid, y_valid),
                    epochs=epochs,
                    batch_size=batch_size,
                    shuffle=False)

    # Extract the specific loss keys for class_predictions
    class_train_loss_key = 'class_loss'
    class_val_loss_key = 'val_class_loss'

    # Plot the class_predictions losses
    plt.plot(history.history[class_train_loss_key], label='Class Train Loss')
    plt.plot(history.history[class_val_loss_key], label='Class Validation Loss')

    # Add labels and legend
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training and Validation Losses Class Prediction')
    plt.show()

    y_pred = LSTM_base.predict(X_test)
    y_pred = np.argmax(y_pred, axis=-1)

    return y_test, y_pred






def extract_valid_windows_v2(df, date_col, input_window, target_window, input_columns, target_columns):
    """
    Extracts valid windows from a time series DataFrame for LSTM training.
    
    Args:
        df (pd.DataFrame): The time series DataFrame with a datetime column.
        date_col (str): The name of the datetime column.
        input_window (int): The number of timesteps for the input sequence.
        target_window (int): The number of timesteps for the target sequence.
        input_columns (list of str): List of column names to include in the input data.
        target_columns (list of str): List of column names to include in the target data.
        
    Returns:
        inputs (list of np.ndarray): List of valid input sequences.
        targets (list of np.ndarray): two values
    """
    # Sort by the datetime column to ensure the time series is ordered
    df = df.sort_values(by=date_col).reset_index(drop=True)
    
    # Ensure the datetime column is in pandas datetime format
    df[date_col] = pd.to_datetime(df[date_col])
    
    # Identify valid consecutive rows (1-hour apart)
    time_diffs = df[date_col].diff().dt.total_seconds()
    valid_indices = time_diffs == 3600  # 1 hour = 3600 seconds
    
    # Mark valid sequences
    valid_sequence_flags = valid_indices | valid_indices.shift(-1, fill_value=False)
    df = df[valid_sequence_flags].reset_index(drop=True)

    # Prepare inputs and targets
    inputs, targets = [], []
    total_window = input_window + target_window

    for i in range(len(df) - total_window + 1):
        # Extract a potential window of size `total_window`
        window = df.iloc[i:i+total_window]
        
        # Check if all rows in the window are 1-hour apart
        if (window[date_col].diff().dt.total_seconds()[1:] == 3600).all():
            # Split into input and target based on specified columns
            input_data = window.iloc[:input_window][input_columns].values
            target_data = window.iloc[input_window:][target_columns].values
            # jsut a simple differnces and sign for now
            differences = target_data[-1, :] - target_data[0, :]
            differences = custom_sign(differences)
            inputs.append(input_data)
            targets.append(differences)

    return np.array(inputs), np.array(targets)




# Custom sign function
def custom_sign(x):
    return np.where(x > 0, 1, np.where(x == 0, 0, 2))



def rate_growth(data, prediction, threshold_factor=0.1):
    # Calculate changes for data
    changes_data = data[:, 23, 0] - data[:, 0, 0]
    std_data = np.std(changes_data)
    threshold_data = threshold_factor * std_data

    # Classify growth for data
    growth_data = []
    for change in changes_data:
        if change > threshold_data:
            growth_data.append(1)
        elif change < -threshold_data:
            growth_data.append(-1)
        else:
            growth_data.append(0)

    # Calculate changes for prediction
    changes_prediction = prediction[:, 23] - prediction[:, 0]
    std_prediction = np.std(changes_prediction)
    threshold_prediction = threshold_factor * std_prediction

    # Classify growth for prediction
    growth_prediction = []
    for change in changes_prediction:
        if change > threshold_prediction:
            growth_prediction.append(1)
        elif change < -threshold_prediction:
            growth_prediction.append(-1)
        else:
            growth_prediction.append(0)

    return growth_data, growth_prediction

def classification_metrics(y_true, y_pred, printed = True):

     # Calculate metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='macro', zero_division=0)
    recall = recall_score(y_true, y_pred, average='macro', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
    
    if printed == True:
        # Calculate the confusion matrix
        conf_matrix = confusion_matrix(y_true, y_pred)
        # Visualization of the confusion matrix using Seaborn
        plt.figure(figsize=(8, 6))
        sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=[ '0', '1', '2'], yticklabels=[ '0', '1', '2'])
        plt.xlabel('Predicted Labels')
        plt.ylabel('True Labels')
        plt.title('Confusion Matrix')
        plt.show()
        # Printing classification metrics
        print("Classification Metrics:")
        print("Accuracy: {:.2f}".format(accuracy))
        print("Precision: {:.2f}".format(precision))
        print("Recall: {:.2f}".format(recall))
        print("F1 Score: {:.2f}".format(f1))
        
        # Detailed classification report
        print("\nDetailed Classification Report:")
        print(classification_report(y_true, y_pred, target_names=[ '0', '1', '2']))

    return accuracy, precision, recall, f1


import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout, Masking, Reshape, Layer, Lambda, Concatenate, LayerNormalization, MultiHeadAttention
from tensorflow.keras.models import Model
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, classification_report
from tensorflow.keras.optimizers import Adadelta
from keras.layers import BatchNormalization

def train_v1(inputs, outputs, test_size=0.2, valid_size=0.25, epochs=100, batch_size=200, d1=0.1, d2 = 0.05, cell_size = 80, details=True):
    # Clearing the TensorFlow session to ensure the model starts with fresh weights and biases
    tf.keras.backend.clear_session()
    n_classes = 3

    cell_size_1 = cell_size
    cell_size_2 = cell_size_1//2

    # Splitting the data into train+validation and test sets
    X_train_valid, X_test, y_train_valid, y_test = train_test_split(
        inputs, outputs, test_size=test_size)

    # Splitting the train+validation set into separate training and validation sets
    X_train, X_valid, y_train, y_valid = train_test_split(
        X_train_valid, y_train_valid, test_size=valid_size)

    # Model definition
    inputs = Input(shape=(X_train.shape[1], X_train.shape[2]))
    Lstm_layer_1 = LSTM(cell_size_1, return_sequences=True, stateful=False)(inputs)
    Batch_norm_1 = BatchNormalization()(Lstm_layer_1)
    Dropout_layer_1 = Dropout(d1)(Batch_norm_1)
    Lstm_layer_2 = LSTM(cell_size_2, return_sequences=False, stateful=False)(Dropout_layer_1)  # just halved
    Batch_norm_2 = BatchNormalization()(Lstm_layer_2)
    Drouput_layer_2 = Dropout(d2)(Batch_norm_2)
    predictions = Dense(n_classes, activation='softmax')(Drouput_layer_2)
    LSTM_base = Model(inputs=inputs, outputs=predictions)

  
    # optimizer
    optimizer = Adadelta(
    learning_rate=1.0,
    rho=0.8,
    epsilon=1e-7)      # Default , to prevent division by zero)

    LSTM_base.compile(
        optimizer=optimizer,
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy'])

    # Training the model
    history = LSTM_base.fit(x=X_train, y=y_train,
                    validation_data=(X_valid, y_valid),
                    epochs=epochs,
                    batch_size=batch_size,
                    shuffle=False,
                    verbose=0)
 
    if details == True:
        LSTM_base.summary()
        fig, ax1 = plt.subplots()

        # Plot losses on the primary y-axis
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss', color='tab:red')
        ax1.plot(history.history['loss'], label='Train Loss', color='red', linestyle='-')
        ax1.plot(history.history['val_loss'], label='Validation Loss', color='red', linestyle='--')
        ax1.tick_params(axis='y', labelcolor='tab:red')

        # Create a second y-axis for accuracy
        ax2 = ax1.twinx()
        ax2.set_ylabel('Accuracy', color='tab:blue')
        ax2.plot(history.history['accuracy'], label='Train Accuracy', color='blue', linestyle='-')
        ax2.plot(history.history['val_accuracy'], label='Validation Accuracy', color='blue', linestyle='--')
        ax2.tick_params(axis='y', labelcolor='tab:blue')

        # Combine legends from both axes
        fig.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=2)  # Legend outside the plot

        plt.title('Model Accuracy and Loss')
        plt.tight_layout()  # Adjust layout to avoid clipping
        plt.show()


    y_pred = LSTM_base.predict(X_test)
    y_pred = np.argmax(y_pred, axis=-1)

    return y_test, y_pred