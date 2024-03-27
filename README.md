---

# DataGenie-Hackathon-CSVBot ( Basic Design )

## Overview
The `DataGenie-Hackathon-CSVBot` is a Streamlit web application that serves as a conversational question-answering bot. It leverages both SQL databases and OpenAI language models (LLMs) to provide responses to user queries. The bot can interact with users in a chat-like interface, storing conversation history and utilizing memory to improve response efficiency.

## Features
- **Interactive Chat Interface**: Provides a chat-like interface for users to input queries and receive responses.
- **Conversation History**: Maintains a history of previous questions and responses for reference.
- **Memory Management**: Stores previous Q&A pairs to improve response efficiency and recall.
- **Integration with SQL Databases**: Connects to PostgreSQL databases to retrieve information based on user queries.
- **OpenAI Language Model Integration**: Utilizes OpenAI language models for natural language understanding and generation.

## Usage
1. **Environment Setup**: Ensure Python environment is set up with necessary dependencies, including Streamlit, psycopg2, OpenAI, and LangChain.
   
2. **Running the Application**: Execute the script `mile-1-2.py` to start the Streamlit web application.

3. **Chat Interface**: Interact with the bot by typing queries into the text area provided. Press Enter to submit the query and view the bot's response.

## Dependencies
- Python 3.x
- Streamlit library (`streamlit`)
- psycopg2 library (`psycopg2`)
- LangChain library (`langchain`)
- OpenAI library (`openai`)
- Other dependencies: `dotenv`

## Configuration
- **Database Connection**: Modify the PostgreSQL database connection parameters (`hostname`, `database`, `username`, `pwd`, `port_id`) as per your database setup.
- **OpenAI Model**: Customize the OpenAI language model parameters, such as temperature, as needed.

## Example
```bash
streamlit run mile-1-2.py
```

## Notes
- Ensure that you have installed all the required libraries and dependencies before running the application.
- The bot may require fine-tuning or adjustments to achieve optimal performance based on specific use cases and data.

## Preview
- ![image](https://github.com/barath-sk17/DataGenie-LLM/assets/127032804/ae1ecade-ba0a-4e20-9b74-d09760758709)


---

# Conversational Question-Answering System with CSV Data and FAISS

## Overview
This Python script `mistral.py` implements a conversational question-answering system using LangChain library components such as CSV data loader, text splitter, Hugging Face embeddings, and FAISS vector store. The system allows users to input queries and receive responses based on the CSV data provided.

## Features
- **CSV Data Loading**: Utilizes the `CSVLoader` from LangChain to load CSV data.
- **Text Chunking**: Splits the loaded text data into smaller chunks using the `RecursiveCharacterTextSplitter`.
- **Embedding Generation**: Utilizes Hugging Face embeddings to convert text chunks into dense embeddings.
- **Vector Store Creation**: Creates a vector store using FAISS to efficiently store and query the embeddings.
- **Conversational Retrieval**: Implements a conversational question-answering system using LangChain's CTransformers LLMS (Language Model) and the FAISS vector store for similarity search.

## Usage
1. **CSV Data Preparation**: Prepare the CSV data file (`output_1.csv`) containing the text information for question-answering.
   
2. **Python Environment Setup**: Set up a Python environment with the required dependencies, including LangChain, Hugging Face Transformers, and FAISS.

3. **Running the Script**: Execute the script `mistral.py`. It will prompt you to input a query, and then it will provide a response based on the CSV data and FAISS vector store.

## Dependencies
- Python 3.x
- LangChain library (`langchain`)
- Hugging Face Transformers library (`transformers`)
- FAISS library (`faiss-cpu` or `faiss-gpu`)
- Other dependencies: `csv`, `os`, `sys`

## Configuration
- **CSV File Path**: Modify the `file_path` variable to specify the path to the CSV file.
- **FAISS Vector Store Path**: Adjust the `DB_FAISS_PATH` variable to set the path for saving the FAISS vector store.
- **LLM Model**: Customize the LLM model path (`CSV_BOT/finallllllllllll/models/mistral-7b-instruct-v0.2.Q6_K.gguf`) and other parameters as needed.

## Example
```bash
python mistral.py
```
---

# Conversational Question-Answering System with CSV Data

## Overview
This Python script `llama.py` demonstrates a conversational question-answering system integrated with CSV data. It leverages various components from the LangChain library to load CSV data, preprocess text, generate embeddings, and perform similarity search for answering questions.

## Features
- **CSV Data Loading**: Utilizes the `CSVLoader` from LangChain to load CSV data.
- **Text Chunking**: Splits the loaded text data into smaller chunks for efficient processing using the `RecursiveCharacterTextSplitter`.
- **Embedding Generation**: Utilizes Hugging Face embeddings to convert text chunks into dense embeddings.
- **Vector Store Creation**: Creates a vector store using FAISS to store and query the embeddings efficiently.
- **Conversational Retrieval**: Implements a conversational question-answering system using LangChain's LLM (Language Model) and the vector store for similarity search.

## Usage
1. **CSV Data**: Prepare the CSV data file (`output_1.csv`) containing the text information for question-answering.
   
2. **Python Environment**: Set up a Python environment with the required dependencies, including LangChain, Hugging Face Transformers, and FAISS.

3. **Running the Script**: Execute the script `llama.py`. It will prompt you to input a query, and then it will provide a response based on the CSV data.

## Dependencies
- Python 3.x
- LangChain library (`langchain`)
- Hugging Face Transformers library (`transformers`)
- FAISS library (`faiss-cpu` or `faiss-gpu`)
- Other dependencies: `csv`, `os`, `sys`

## Configuration
- **CSV File Path**: Modify the `file_path` variable to specify the path to the CSV file.
- **Model**: Customize the LLM model path (`models/llama-2-7b-chat.ggmlv3.q8_0.bin`) and other parameters as needed.
- **Vector Store Path**: Adjust the `DB_FAISS_PATH` variable to set the path for saving the FAISS vector store.

## Run-Command
```bash
python llama.py
```
---

# CSV to PostgreSQL Data Importer

## Overview
This Python script `script_ai.py` is designed to import data from a CSV file into a PostgreSQL database. It automates the process of creating a table based on the CSV file's structure and inserting the data into the database.

## Features
- Parses the structure of the CSV file to generate a corresponding PostgreSQL table schema dynamically.
- Handles different data types (integer, float, string) in the CSV file and creates appropriate table columns.
- Inserts data from the CSV file into the PostgreSQL table using parameterized queries to prevent SQL injection.
- Provides flexibility to customize the input CSV file path, target table name, and PostgreSQL connection parameters.

## Usage
1. **Input CSV File**: Specify the path to the input CSV file using the `file_path` variable.
   
2. **Target Table Name**: Define the name of the table where the data will be imported. Modify the `table_name` variable accordingly.

3. **PostgreSQL Connection Parameters**: Set the connection parameters for your PostgreSQL database (hostname, database name, username, password, port) using the respective variables (`hostname`, `database`, `username`, `pwd`, `port_id`).

4. **Running the Script**: Execute the script `script_ai.py`. It will read the input CSV file, dynamically generate the table schema, create the table in the PostgreSQL database, and insert the data.

## Run-Command
```bash
python script_ai.py
```

---

# CSV Partitioning Utility

## Overview
This Python script `csv-conv.py` is designed to partition a large CSV file into smaller files with a specified number of rows per file. This can be useful for breaking down large datasets into more manageable chunks, especially when dealing with memory constraints or when processing the data in parallel.

## Usage
1. **Input File**: The script expects a CSV file as input. Make sure to specify the path to the input CSV file using the `input_file` variable.
   
2. **Output Folder**: Define the folder where the partitioned CSV files will be saved. Specify the path to the output folder using the `output_folder` variable.

3. **Rows per File**: Optionally, you can specify the number of rows per file for partitioning. By default, each output file will contain 1000 rows, but you can adjust this value by modifying the `rows_per_file` parameter in the `partition_csv` function.

4. **Running the Script**: Simply execute the script `csv-conv.py`. It will read the input CSV file, partition it into smaller CSV files based on the specified number of rows per file, and save them in the output folder.

## Run-Command
```bash
python csv-conv.py
```

## Output
Suppose you have a large CSV file named `SalesUseCase-V1.csv`. After running the script, it will create a folder named `Sales-Datagen` (if it doesn't exist already) and save the partitioned CSV files inside it. The output files will be named `output_1.csv`, `output_2.csv`, etc., each containing the specified number of rows.

