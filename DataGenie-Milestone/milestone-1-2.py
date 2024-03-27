from dotenv import load_dotenv
from streamlit_chat import message
from langchain_experimental.sql import SQLDatabaseChain
from langchain import OpenAI, SQLDatabase
import psycopg2
import streamlit as st

load_dotenv()
from sqlalchemy import create_engine, text

# Streamlit setup
menu = st.sidebar.selectbox("Choose an Option", ["DataGenie-Hackathon-CSVBot"])
st.title("DataGenie-Hackathon-CSVBot")
st.info("Chat Below")

#input_text = st.text_area("Enter your query")


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


def conversational_chat(query):
    result = db_chain.run(query)
    st.session_state['history'].append(result)
    return result

if 'history' not in st.session_state:
    st.session_state['history'] = []

if 'generated' not in st.session_state:
    st.session_state['generated'] = ["Hello ! Ask me anything about the dataset"]

if 'past' not in st.session_state:
    st.session_state['past'] = ["Hey ! 👋"]

#container for the chat history
response_container = st.container()
#container for the user's text input
container = st.container()

with container:
    with st.form(key='my_form', clear_on_submit=True):
        
        user_input = st.text_input("Query:", placeholder="Talk to your csv", key='input')
        submit_button = st.form_submit_button(label='Send')
        
    if submit_button and user_input:
        output = conversational_chat(user_input)
        
        st.session_state['past'].append(user_input)
        st.session_state['generated'].append(output)

if st.session_state['generated']:
    with response_container:
        for i in range(len(st.session_state['generated'])):
            message(st.session_state["past"][i], is_user=True, key=str(i) + '_user', avatar_style="big-smile")
            message(st.session_state["generated"][i], key=str(i), avatar_style="thumbs")
