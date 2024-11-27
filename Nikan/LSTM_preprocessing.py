import numpy as np
import pandas as pd
import mysql.connector
from mysql.connector import Error

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
    SELECT clb.*, cp.*
    FROM crypto_lending_borrowing clb
    JOIN crypto_price cp 
        ON clb.crypto_symbol = cp.crypto_symbol
        AND clb.date = cp.date
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
def calculate_iqr_bounds(series):
    Q1 = series.quantile(0.25)
    Q3 = series.quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = max(Q1 - 1.5 * IQR, 0)
    upper_bound = Q3 + 1.5 * IQR
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