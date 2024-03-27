from dotenv import load_dotenv
from langchain_experimental.sql import SQLDatabaseChain
from langchain import OpenAI, SQLDatabase
import psycopg2
import streamlit as st

load_dotenv()
from sqlalchemy import create_engine, text

# Session state for storing previous interaction
if 'prev_question' not in st.session_state:
    st.session_state['prev_question'] = None
if 'prev_response' not in st.session_state:
    st.session_state['prev_response'] = None


# Streamlit setup
menu = st.sidebar.selectbox("Choose an Option", ["DataGenie-Hackathon-CSVBot"])
st.title("DataGenie-Hackathon-CSVBot")
st.info("Chat Below")

if st.session_state['prev_question'] and st.session_state['prev_response']:
    st.write("Previous Question:", st.session_state['prev_question'])
    st.write("Previous Response:", st.session_state['prev_response'])

input_text = st.text_area("Enter your query")

# Session state for conversation history and memory
if "past_qa" not in st.session_state:
    st.session_state["past_qa"] = {}  # Dictionary to store Q&A pairs

# PostgreSQL database connection setup
hostname = 'localhost'
database = 'datagen'
username = 'postgres'
pwd = 'barathsk617'
port_id = 5432

# Construct the connection string for PostgreSQL
db = SQLDatabase.from_uri(f"postgresql://{username}:{pwd}@{hostname}:{port_id}/{database}")

# Initialize OpenAI
llm = OpenAI(temperature=0)
db_chain = SQLDatabaseChain(llm=llm, database=db, verbose=True)

# Execute the query if the input is not empty
if input_text:
    try:

        st.session_state['prev_question'] = input_text

        # Check if question already exists in memory
        if input_text in st.session_state["past_qa"]:
            answer = st.session_state["past_qa"][input_text]
            st.session_state['prev_response'] = answer
            st.success(f"Found answer in memory: {answer}")  # Display from memory
        else:
            # Call the original logic for database or LLM processing
            result = db_chain.run(input_text)

            # Update memory with new Q&A pair
            st.session_state["past_qa"][input_text] = result
            st.session_state['prev_response'] = result
            st.success(result)  # Display from database/LLM
        
            

    except Exception as e:
        st.error(f"An error occurred: {e}")
