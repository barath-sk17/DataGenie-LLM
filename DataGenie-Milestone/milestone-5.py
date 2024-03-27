from langchain.chat_models import ChatOpenAI
from dotenv import load_dotenv
from langchain.chains import QAGenerationChain
from langchain.text_splitter import TokenTextSplitter
from langchain.docstore.document import Document
from langchain.document_loaders import PyPDFLoader
from langchain.prompts import PromptTemplate
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma, FAISS
from langchain.chains.summarize import load_summarize_chain
from langchain.chains import RetrievalQA
from dotenv import load_dotenv
from langchain_experimental.sql import SQLDatabaseChain
from langchain import OpenAI, SQLDatabase
from streamlit_chat import message
import psycopg2
import streamlit as st
import os
from uptrain import EvalLLM, Evals, CritiqueTone
import json
import seaborn as sns
import matplotlib.pyplot as plt

# Load environment variables
from sqlalchemy import create_engine, text

from lida import Manager, TextGenerationConfig , llm  
import os
import openai as openai_module
from PIL import Image
from io import BytesIO
import base64

load_dotenv()
openai_module.api_key = os.getenv('OPENAI_API_KEY')

def base64_to_image(base64_string):
    # Decode the base64 string
    byte_data = base64.b64decode(base64_string)
    
    # Use BytesIO to convert the byte data to image
    return Image.open(BytesIO(byte_data))


lida = Manager(text_gen = llm("openai"))
textgen_config = TextGenerationConfig(n=1, temperature=0.5, model="gpt-3.5-turbo-0301", use_cache=True)


# PostgreSQL database connection setup
hostname = 'localhost'
database = 'datagen'
username = 'postgres'
pwd = 'barathsk617'
port_id = 5432


menu = st.sidebar.selectbox("Choose an Option", ["DataGenie-Hackathon-CSVBot","Anti Hallinucation & Response Evaluation","Question based Graph","Upload CSV for Graph Summarize","Upload CSV & Visualizse"])
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

elif menu == "Question based Graph":
    llm = OpenAI(temperature=0)
    db1 = SQLDatabase.from_uri(f"postgresql://{username}:{pwd}@{hostname}:{port_id}/{database}")
    st.subheader("Query your Data to Generate Graph")
    query = st.text_input("Enter a query about top countries based on rank:")
    if query:
        db_chain = SQLDatabaseChain(llm=llm, database=db1, verbose=True, return_intermediate_steps=True)
        response = db_chain(query)
        intermediateSteps=response["intermediate_steps"][3]
        
        data_list = eval(intermediateSteps)

        fig, ax = plt.subplots(figsize=(10, 6))
        # Extract country names and hunger scores
        x_axis = [x for x, _ in data_list]
        y_axis = [y for _, y in data_list]

        fig, ax = plt.subplots()  # Create a figure and axes for better control
        sns.barplot(x=x_axis, y=y_axis, ax=ax)  # Use the axes for plotting
        plt.xticks(rotation=45)

        st.pyplot(fig)

elif menu == "Upload CSV for Graph Summarize":
    st.subheader("Summarization of your Data")
    file_uploader = st.file_uploader("Upload your CSV", type="csv")
    if file_uploader is not None:
        path_to_save = "process.csv"
        with open(path_to_save, "wb") as f:
            f.write(file_uploader.getvalue())
        summary = lida.summarize("process.csv", summary_method="default", textgen_config=textgen_config)
        st.write(summary)
        goals = lida.goals(summary, n=2, textgen_config=textgen_config)
        for goal in goals:
            st.write(goal)
        i = 0
        library = "seaborn"
        textgen_config = TextGenerationConfig(n=1, temperature=0.2, use_cache=True)
        charts = lida.visualize(summary=summary, goal=goals[i], textgen_config=textgen_config, library=library)  
        img_base64_string = charts[0].raster
        img = base64_to_image(img_base64_string)
        st.image(img)
        


        
elif menu == "Upload CSV & Visualizse":
    st.subheader("Upload CSV & Ask Quses!")
    file_uploader = st.file_uploader("Upload your CSV", type="csv")
    if file_uploader is not None:
        path_to_save = "process_query.csv"
        with open(path_to_save, "wb") as f:
            f.write(file_uploader.getvalue())
        text_area = st.text_area("Query your Data to Generate Graph")
        if st.button("Generate Graph"):
            if len(text_area) > 0:
                st.info("Your Query: " + text_area)
                lida = Manager(text_gen = llm("openai")) 
                textgen_config = TextGenerationConfig(n=1, temperature=0.2, use_cache=True)
                summary = lida.summarize("process_query.csv", summary_method="default", textgen_config=textgen_config)
                user_query = text_area
                charts = lida.visualize(summary=summary, goal=user_query, textgen_config=textgen_config)  
                charts[0]
                image_base64 = charts[0].raster
                img = base64_to_image(image_base64)
                st.image(img)



