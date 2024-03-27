import csv
import psycopg2

hostname = 'localhost'
database = 'datagen'
username = 'postgres'
pwd = 'barathsk617'
port_id = 5432

conn = None
cur = None

def parse_insert_statement(insert_statement):
    # Extracting table name
    table_name_start = insert_statement.find("INTO ") + 5
    table_name_end = insert_statement.find("(", table_name_start)
    table_name = insert_statement[table_name_start:table_name_end].strip()

    # Extracting column names and values
    columns_start = insert_statement.find("(", table_name_end) + 1
    columns_end = insert_statement.find(")", columns_start)
    columns_str = insert_statement[columns_start:columns_end]
    columns = [col.strip().split()[0] for col in columns_str.split(",")]

    values_start = insert_statement.find("VALUES (") + 8
    values_end = insert_statement.find(");")
    values_str = insert_statement[values_start:values_end]
    values = [val.strip() for val in values_str.split(",")]

    return table_name, columns, values

def generate_create_statement(table_name, columns, values):
    create_statement = f"CREATE TABLE {table_name} (\n"
    for column, value in zip(columns, values):
        data_type = "VARCHAR(255)" if value.startswith("'") else "INTEGER"
        create_statement += f'    {column} {data_type},\n'
    create_statement = create_statement.rstrip(",\n") + "\n);"
    return create_statement

def read_csv(file_path):
    with open(file_path, 'r', newline='', encoding='utf-8') as file:
        reader = csv.DictReader(file)
        rows = [row for row in reader]
    return rows

def determine_value_type(value):
    try:
        float_value = float(value)
        int_value = int(float_value)
        if float_value == int_value:
            return int
        else:
            return float
    except ValueError:
        return str

def convert_to_postgres_queries(rows, table_name):
    queries = []
    for row in rows:
        columns = ', '.join([f'"{key.replace(" ", "_")}"' for key in row.keys()])  # Replace spaces with underscores
        values = []
        for value in row.values():
            value_type = determine_value_type(value)
            if value_type == int:
                values.append(value)
            elif value_type == float:
                values.append(value)
            else:
                values.append("'" + value.replace("'", "''") + "'")
        values_template = ', '.join(map(str, values))
        query = f"INSERT INTO {table_name} ({columns}) VALUES ({values_template});"
        queries.append(query)
    return queries, query

def main():
    file_path = 'Sales-Datagen.csv/output_1.csv'  # Replace 'data.csv' with the path to your CSV file
    table_name = 'CustomerData'  # Replace 'your_table_name' with the actual table name in your PostgreSQL database
    
    rows = read_csv(file_path)
    postgres_queries, q = convert_to_postgres_queries(rows, table_name)
    
    complete_script = ""
    
    # Parse INSERT statement
    table_name, columns, values = parse_insert_statement(q)

    # Generate CREATE statement
    create_statement = generate_create_statement(table_name, columns, values)
    
    # Append CREATE TABLE statement
    complete_script += create_statement + "\n\n"
    
    # Append INSERT INTO statements
    for query in postgres_queries:
        complete_script += query + "\n"
    

    
    try:
        conn = psycopg2.connect(
            host=hostname,
            dbname=database,
            user=username,
            password=pwd,
            port=port_id
        )

        cur = conn.cursor()

        create_script = complete_script
        cur.execute(create_script)
        conn.commit()

    except Exception as error:
        print(error)
    finally:
        if cur is not None:
            cur.close()
        if conn is not None:
            conn.close()
    

if __name__ == "__main__":
    
    main()
