from dotenv import load_dotenv
from langchain_experimental.sql import SQLDatabaseChain
from langchain import OpenAI, SQLDatabase
from streamlit_chat import message
import psycopg2
import streamlit as st
import os
from uptrain import EvalLLM, Evals, CritiqueTone
import json

# Load environment variables
load_dotenv()
from sqlalchemy import create_engine, text

# PostgreSQL database connection setup
hostname = 'localhost'
database = 'datagen'
username = 'postgres'
pwd = 'barathsk617'
port_id = 5432

menu = st.sidebar.selectbox("Choose an Option", ["DataGenie-Hackathon-CSVBot","Anti Hallinucation & Response Evaluation"])
# Streamlit setup

if menu == "DataGenie-Hackathon-CSVBot":
    st.title("DataGenie-Hackathon-CSVBot")
    st.info("Chat Below")

    #input_text = st.text_area("Enter your query")

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


elif menu == "Anti Hallinucation & Response Evaluation":

    st.title("Anti Hallinucation & Response Evaluation")
    st.info("Chat Below")
    
    # Session state for storing previous interaction
    if 'prev_question' not in st.session_state:
        st.session_state['prev_question'] = None
    if 'prev_response' not in st.session_state:
        st.session_state['prev_response'] = None

    if st.session_state['prev_question'] and st.session_state['prev_response']:
        st.write("Previous Question:", st.session_state['prev_question'])
        st.write("Previous Response:", st.session_state['prev_response'])

    
    input_text = st.text_area("Enter your query")

    if "past_qa" not in st.session_state:
        st.session_state["past_qa"] = {}

    # Construct the connection string for PostgreSQL
    db = SQLDatabase.from_uri(f"postgresql://{username}:{pwd}@{hostname}:{port_id}/{database}")

    # Initialize OpenAI and SQLDatabaseChain (replace with your OpenAI API key)
    llm = OpenAI(temperature=0, openai_api_key=os.getenv("OPENAI_API_KEY"))
    db_chain = SQLDatabaseChain(llm=llm, database=db, verbose=True)

    # Function to evaluate LLM response using UpTrain
    def evaluate_llm_response(question, context, response):
        data = [{
            'question': question,
            'context': context,
            'response': response
        }]

        eval_llm = EvalLLM(openai_api_key=os.getenv("OPENAI_API_KEY"))
        results = eval_llm.evaluate(
            data=data,
            checks=[Evals.CONTEXT_RELEVANCE, Evals.FACTUAL_ACCURACY, Evals.RESPONSE_RELEVANCE, CritiqueTone(persona="teacher")]
        )
        return results

    # Main execution loop
    if input_text:
        try:
            st.session_state['prev_question'] = input_text

            context_measured = '''
            The customerdata table stores information about a single day's transactions (December 15th, 2020). It includes details about the customer (age range, gender, income group), how they found out about the product (marketing channel), what they purchased (product category, subcategory, brand, name), how much it cost (unit cost, total cost, discount), how many they bought (number of units), how it was shipped (shipping cost, delivery time), and how they paid for it (payment method). While some data has a wider range (e.g., unit cost: $9 to $2324), most values fall within specific ranges (e.g., number of units purchased: 1 to 21). The customer data itself is limited to a specific set of states (PA, CA, TX, FL, IL, NC, OH, NY, GA, MI) and shows a bias towards a certain age group (26-35 year olds).
            This dataset offers a snapshot of customer transactions on December 15th, 2020. While the sample size is limited to 500 records from ten states, it reveals some interesting trends. The data suggests a young customer base, with a majority falling between 26-35 years old. These customers are tech-savvy, utilizing a variety of channels like PPC and Apps to make purchases. Popular product categories include Electronics, Clothing, and Home goods, with established brands like HP, Revlon, and Tempur-Pedic being well-represented. The analysis of unit costs, total costs, and profit margins would provide further insights into customer behavior and spending habits. It would be interesting to see how these trends compare to other days throughout the year, revealing seasonal variations in customer preferences.
            '''

            # Check if question already exists in memory
            if input_text in st.session_state["past_qa"]:
                answer = st.session_state["past_qa"][input_text]
                st.session_state['prev_response'] = answer
                st.success(f"Found answer in memory: {answer}")  # Display from memory

                if answer:  # Check if result is not empty
                    evaluation = evaluate_llm_response(input_text, context=context_measured, response=answer)
                    st.write("UpTrain Evaluation:")
                    st.json(evaluation)
            else:
                # Call the original logic for database or LLM processing
                result = db_chain.run(input_text)

                # Update memory with new Q&A pair
                st.session_state["past_qa"][input_text] = result
                st.session_state['prev_response'] = result
                st.success(result)  # Display from database/LLM

                if result:  # Check if result is not empty
                    evaluation = evaluate_llm_response(input_text, context=context_measured, response=result)
                    st.write("UpTrain Evaluation:")
                    st.json(evaluation)
            
            
        except Exception as e:
            st.error(f"An error occurred: {e}")
