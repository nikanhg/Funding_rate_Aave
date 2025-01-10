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

# Function to check if tables have data
def check_table_data(connection, table_name):
    cursor = connection.cursor()

    try:
        query = f"SELECT COUNT(*) FROM {table_name}"
        cursor.execute(query)
        count = cursor.fetchone()[0]

        if count > 0:
            print(f"Table '{table_name}' has {count} rows.")
        else:
            print(f"Table '{table_name}' is empty.")
    except Error as e:
        print(f"Error while checking table data: {e}")
    finally:
        cursor.close()

# Function to check if date formats are consistent between two tables
def check_date_format(connection):
    cursor = connection.cursor()

    try:
        query_lending = "SELECT DATA_TYPE FROM INFORMATION_SCHEMA.COLUMNS WHERE TABLE_NAME = 'crypto_lending_borrowing' AND COLUMN_NAME = 'date'"
        query_price = "SELECT DATA_TYPE FROM INFORMATION_SCHEMA.COLUMNS WHERE TABLE_NAME = 'crypto_price' AND COLUMN_NAME = 'date'"

        cursor.execute(query_lending)
        lending_date_type = cursor.fetchone()[0]

        cursor.execute(query_price)
        price_date_type = cursor.fetchone()[0]

        if lending_date_type == price_date_type:
            print(f"Date format is consistent: {lending_date_type}")
        else:
            print(f"Date format mismatch: crypto_lending_borrowing ({lending_date_type}) vs crypto_price ({price_date_type})")
    except Error as e:
        print(f"Error while checking date format: {e}")
    finally:
        cursor.close()

# Function to check if symbols exist in both tables
def check_common_symbols(connection):
    cursor = connection.cursor()

    try:
        query = """
        SELECT DISTINCT clb.crypto_symbol
        FROM crypto_lending_borrowing clb
        JOIN crypto_price cp ON clb.crypto_symbol = cp.crypto_symbol
        """
        cursor.execute(query)
        common_symbols = cursor.fetchall()

        if len(common_symbols) > 0:
            print(f"Common symbols found in both tables: {[symbol[0] for symbol in common_symbols]}")
        else:
            print("No common symbols found in both tables.")
    except Error as e:
        print(f"Error while checking common symbols: {e}")
    finally:
        cursor.close()



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

# Function to show tables and columns
def query_show(connection):
    cursor = connection.cursor()

    try:
        # 查看数据库中的所有表
        cursor.execute("SHOW TABLES;")
        tables = cursor.fetchall()

        for table in tables:
            table_name = table[0]
            print(f"\nTable: {table_name}")

            # 查看每个表的表头（列名）
            cursor.execute(f"DESCRIBE {table_name};")
            columns = cursor.fetchall()

            for column in columns:
                print(f" - {column[0]} ({column[1]})")
    except Error as e:
        print(f"Error: {e}")
    finally:
        cursor.close()

# Function to close the database connection
def query_quit(connection):
    if connection.is_connected():
        connection.close()
        print("MySQL connection is closed")

# Main function
def main():
    # Connect to MySQL database
    connection = connect_to_database()

    if connection:
        # Show tables and columns
        query_show(connection)

        # Check if tables have data
        check_table_data(connection, 'crypto_lending_borrowing')
        check_table_data(connection, 'crypto_price')

        # Check if date formats are consistent
        check_date_format(connection)

        # Check if symbols exist in both tables
        check_common_symbols(connection)

        # Query merged data
        merged_df = query_merged_crypto_data(connection)

        if merged_df is not None and not merged_df.empty:
            # Display first few rows of the DataFrame
            print("\nMerged DataFrame:")
            print(merged_df.head())

            # Save DataFrame to CSV
            merged_df.to_csv('merged_crypto_data.csv', index=False)
            print("\nMerged data saved to 'merged_crypto_data.csv'")
        else:
            print("\nNo data found after merging.")

        # Close the connection
        query_quit(connection)

# Execute the main function
if __name__ == "__main__":
    main()
