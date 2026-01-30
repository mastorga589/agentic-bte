from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, convert_to_messages
from os.path import dirname, join
from dotenv import load_dotenv
from typing import Literal
from typing_extensions import TypedDict, Annotated
from langgraph.graph import StateGraph, START, MessagesState, END
from langgraph.types import Command
from operator import add
import pandas as pd
import getpass
import os

# Set your OpenAI API key
dotenv_path = join(dirname(__file__), '.env')
load_dotenv(dotenv_path)

if not os.environ.get("OPENAI_API_KEY"): #field to ask for OpenAI API key
    os.environ["OPENAI_API_KEY"] = getpass.getpass("Please enter OpenAI API Key: ")

# Create an LLM-based agent
llm = ChatOpenAI(temperature=0, model="gpt-4o")  # Change model if needed

def get_llm_response(question):
    return llm.invoke(question).content

drugDiseaseDMDB = pd.read_excel('/Users/mastorga/Documents/BTE-LLM/Prototype/logs/25questions_07.15.25.xlsx')

drugDiseaseSet = drugDiseaseDMDB

drugDiseaseSet['gpt4.0_response'] = drugDiseaseSet["question"].apply(get_llm_response)

drugDiseaseSet.to_excel('Prototype/logs/25questions_07.15.25_llm.xlsx')
