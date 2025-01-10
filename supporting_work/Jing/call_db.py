import mysql.connector
from mysql.connector import Error
import pandas as pd

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

# Function to query crypto_price table
def query_crypto_price(connection, symbol, start_date, end_date):
    query = f"""
    SELECT crypto_symbol, date, high, low, close, volume
    FROM crypto_price
    WHERE crypto_symbol = %s AND date BETWEEN %s AND %s
    ORDER BY date ASC
    """
    cursor = connection.cursor()

    try:
        # Execute the query
        cursor.execute(query, (symbol, start_date, end_date))

        # Fetch all results
        results = cursor.fetchall()

        # Convert results to a Pandas DataFrame for easier manipulation
        df = pd.DataFrame(results, columns=['crypto_symbol', 'date', 'high', 'low', 'close', 'volume'])

        return df

    except Error as e:
        print(f"Error: {e}")
        return None
    finally:
        cursor.close()

# Main function
def main():
    # Connect to MySQL database
    connection = connect_to_database()

    if connection:
        # Define the parameters for the query
        crypto_symbol = 'BTCUSDT'
        start_date = '2023-08-01'
        end_date = '2023-08-31'

        # Query the crypto_price table
        df = query_crypto_price(connection, crypto_symbol, start_date, end_date)

        if df is not None:
            # Print the first 5 rows of the DataFrame
            print(df.head())
            print(df.describe())

        # Close the database connection
        connection.close()
        print("MySQL database connection closed")

# Execute the main function
if __name__ == "__main__":
    main()
