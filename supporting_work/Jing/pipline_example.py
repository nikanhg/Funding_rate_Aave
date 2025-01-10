import requests
import pandas as pd
import mysql.connector
from mysql.connector import Error
import calendar

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

# Function to get data from Binance API
def get_binance_klines(symbol, interval, start_time, end_time):
    url = "https://api.binance.com/api/v3/klines"
    params = {
        'symbol': symbol,
        'interval': interval,
        'startTime': start_time,
        'endTime': end_time,
        'limit': 720  # Requesting 720 data points (12 hours)
    }
    all_data = []

    while True:
        response = requests.get(url, params=params)
        data = response.json()

        if not data:
            break

        all_data.extend(data)
        params['startTime'] = data[-1][0] + 1  # Update start time to the last data point

        if params['startTime'] >= end_time:
            break

    # Convert to DataFrame
    df = pd.DataFrame(all_data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume',
                                         'close_time', 'quote_asset_volume', 'number_of_trades',
                                         'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    return df

# Function to store data into MySQL database (crypto_price table)
def store_to_mysql(df, connection, table_name, crypto_symbol):
    cursor = connection.cursor()

    # Step 1: Check if the crypto_symbol exists in crypto_reference table
    check_query = "SELECT crypto_symbol FROM crypto_reference WHERE crypto_symbol = %s"
    cursor.execute(check_query, (crypto_symbol,))
    result = cursor.fetchone()

    # Step 2: If the crypto_symbol does not exist, insert it into the crypto_reference table
    if result is None:
        print(f"{crypto_symbol} not found in crypto_reference, inserting now...")
        insert_reference_query = "INSERT INTO crypto_reference (crypto_symbol, crypto_name) VALUES (%s, %s)"
        cursor.execute(insert_reference_query, (crypto_symbol, crypto_symbol))  # Assuming crypto_name is the same as the symbol

    # Step 3: Insert data into crypto_price table, use ON DUPLICATE KEY UPDATE to prevent duplicates
    insert_query = f"""
    INSERT INTO {table_name} 
    (crypto_symbol, date, high, low, close, volume, market_cap, coin_supply) 
    VALUES (%s, %s, %s, %s, %s, %s, NULL, NULL)
    ON DUPLICATE KEY UPDATE 
    high=VALUES(high), low=VALUES(low), close=VALUES(close), volume=VALUES(volume)
    """

    for row in df.itertuples(index=False):
        cursor.execute(insert_query, (crypto_symbol, row.timestamp, row.high, row.low, row.close, row.volume))

    connection.commit()
    print(f"{len(df)} records successfully inserted into {table_name} table")

# Function to fetch data in batches for a specific month and year
def fetch_and_store_for_month(symbol, connection, table_name, year, month):
    # Calculate the start and end time for the given month and year
    start_time = pd.Timestamp(year=year, month=month, day=1)
    end_day = calendar.monthrange(year, month)[1]  # Get the last day of the month
    end_time = pd.Timestamp(year=year, month=month, day=end_day, hour=23, minute=59, second=59)

    # Convert to Unix timestamps (in milliseconds)
    start_time_ms = int(start_time.timestamp() * 1000)
    end_time_ms = int(end_time.timestamp() * 1000)

    # Fetch data in 12-hour batches (720 minutes)
    current_time_ms = start_time_ms
    while current_time_ms < end_time_ms:
        next_time_ms = min(current_time_ms + 720 * 60 * 1000, end_time_ms)  # 720 minutes (12 hours)

        # Get data for the current time range
        df = get_binance_klines(symbol, '1m', current_time_ms, next_time_ms)

        # Store data in MySQL database
        store_to_mysql(df, connection, table_name, symbol)

        # Update current time and proceed to the next batch
        current_time_ms = next_time_ms

# Main function
def main():
    # Example: Specify the year and month you want to fetch data for
    year = 2023
    month = 8  # August
    
    # Connect to MySQL database
    connection = connect_to_database()

    if connection:
        # Fetch and store data for the specified month and year
        fetch_and_store_for_month('BTCUSDT', connection, 'crypto_price', year, month)

        # Close the database connection
        connection.close()
        print("MySQL database connection closed")

# Execute the main function
if __name__ == "__main__":
    main()
