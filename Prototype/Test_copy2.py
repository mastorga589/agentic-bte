from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, convert_to_messages
from os.path import dirname, join
from dotenv import load_dotenv
from typing import Literal
from typing_extensions import TypedDict, Annotated
from langgraph.graph import StateGraph, START, MessagesState, END
from langgraph.types import Command
from operator import add
from Agent import BTEx
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

drugDiseaseDMDB = pd.read_excel('/Users/mastorga/Documents/BTE-LLM/Prototype/logs/disease <- drugbp/save1_50questions_diseasefromdrugbp_07.22.25.xlsx')

drugDiseaseSet = drugDiseaseDMDB

drugDiseaseSet["BTEx_response (maxresult = 50, k = 10)"] = drugDiseaseSet["question"].apply(lambda q: BTEx(q, maxresults=50, k=10))

drugDiseaseSet.to_excel('/Users/mastorga/Documents/BTE-LLM/Prototype/logs/disease <- drugbp/50m10k_50questions_diseasefromdrugbp_07.22.25.xlsx')
