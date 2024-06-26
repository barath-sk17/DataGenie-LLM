---

# Streamlit Application for Various Data-Related Tasks

## Overview
This Python script implements a Streamlit application with multiple functionalities related to data processing, analysis, and interaction. It leverages various libraries such as LangChain, OpenAI, UpTrain, psycopg2, and Streamlit to provide a user-friendly interface for performing tasks like conversational question-answering, question generation, data summarization, and graph visualization.

## Features
- **DataGenie-Hackathon-CSVBot**: Implements a conversational question-answering system based on CSV data, allowing users to interactively query the dataset and receive responses.
- **Anti Hallucination & Response Evaluation**: Evaluates the responses generated by an AI model (LLM) for contextual relevance, factual accuracy, response relevance, and tone using the UpTrain library.
- **Question Generator**: Generates questions based on the conversation history, helping users prepare for exams or coding tests.
- **Question Based Graph**: Queries the dataset to generate a graph based on user input, visualizing data related to top countries based on rank.
- **Upload CSV for Graph Summarization**: Summarizes the uploaded CSV data and generates goals for further analysis. Visualizes the data using charts (e.g., seaborn).
- **Upload CSV & Visualize**: Allows users to upload a CSV file and query the data to generate a graph based on the user's input.

## Usage
1. **Environment Setup**: Set up a Python environment with the required dependencies listed in the `requirements2.txt` file.
2. **Configuration**: Ensure that the PostgreSQL database connection parameters (`hostname`, `database`, `username`, `pwd`, `port_id`) are correctly configured.
3. **Running the Script**: Execute the script (`streamlit run milestone-5.py`) & (`streamlit run milestone-4.py`) to launch the Streamlit application. Choose the desired option from the sidebar menu to access different functionalities.

## For Experiencing Different Features

```bash
streamlit run milestone-4.py
```
```bash
streamlit run milestone-5.py
```

## Dependencies
- Python 3.x
- LangChain
- Streamlit  
- psycopg2
- UpTrain
- seaborn
- matplotlib
- openai
- lida

Install the dependencies using `pip install -r requirements2.txt`.

## Configuration
- **PostgreSQL Connection**: Configure the PostgreSQL database connection parameters (`hostname`, `database`, `username`, `pwd`, `port_id`) in the script.
- **OpenAI API Key**: Set up an OpenAI API key and ensure it's loaded using the `load_dotenv()` function with a corresponding `.env` file.

## Notes
- Ensure that you have the necessary permissions and access to the PostgreSQL database.
- Customize the script according to your specific dataset and requirements.
- For production use, consider securing sensitive information such as API keys and database credentials.

## Video Preview
- https://drive.google.com/file/d/1ndt6A8R4y3rqpF6qp5N95-0OeITAFJTh/view?usp=sharing

## Preview
- ![image](https://github.com/barath-sk17/DataGenie-LLM/assets/127032804/51a1e9f8-5986-4066-a1a9-2b76d03150f8)
- ![image](https://github.com/barath-sk17/DataGenie-LLM/assets/127032804/c62ee932-8ee2-468b-915e-63e78a2f764c)
- ![image](https://github.com/barath-sk17/DataGenie-LLM/assets/127032804/eb70c7fd-7836-44f5-8b79-3103003c7661)
- ![image](https://github.com/barath-sk17/DataGenie-LLM/assets/127032804/40b49f5f-1da2-462b-b175-1c7be1183e04)
- ![image](https://github.com/barath-sk17/DataGenie-LLM/assets/127032804/f5c6694f-4f2b-4838-b87b-faf780a538f6)
- ![image](https://github.com/barath-sk17/DataGenie-LLM/assets/127032804/26b323b4-253e-485c-96a5-98ed6e71caa1)
- ![image](https://github.com/barath-sk17/DataGenie-LLM/assets/127032804/cb2a1ef3-8565-4b51-9d9e-9c34baa7da35)
- ![image](https://github.com/barath-sk17/DataGenie-LLM/assets/127032804/7478b54f-30eb-4afe-b9bd-60ab5e6fa555)
- ![image](https://github.com/barath-sk17/DataGenie-LLM/assets/127032804/7fde0718-8961-4896-87ea-b3658c13adc8)








