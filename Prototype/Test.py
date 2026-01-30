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
import random

# Set your OpenAI API key
dotenv_path = join(dirname(__file__), '.env')
load_dotenv(dotenv_path)

if not os.environ.get("OPENAI_API_KEY"): #field to ask for OpenAI API key
    os.environ["OPENAI_API_KEY"] = getpass.getpass("Please enter OpenAI API Key: ")

# Create an LLM-based agent
llm = ChatOpenAI(temperature=0, model="gpt-4o")  # Change model if needed

def get_llm_response(question):
    return llm.invoke(question).content

def get_distractors(df, correct_row, drug_col='drug_name', disease_col='disease_name', bp_col='bp_name', n=4):
    correct_drug = correct_row[drug_col]
    correct_disease = correct_row[disease_col]
    correct_bp = correct_row[bp_col]
    
    # Filter out drugs that match either the disease or biological process
    distractors = df[
        (df[disease_col] != correct_disease) &
        (df[bp_col] != correct_bp) &
        (df[drug_col] != correct_drug)
    ]
    
    # Ensure enough distractors
    if len(distractors) < n:
        return []
    
    sampled = distractors.sample(n=n)[drug_col].tolist()
    sampled += [correct_drug]
    random.shuffle(sampled)

    separator = "\n- "
    choices = separator.join(sampled)
    
    return choices


drugDiseaseDMDB = pd.read_csv('/Users/mastorga/Documents/BTE-LLM/Prototype/data/DMDB_go_bp_filtered_jjoy_05_08_2025.csv')

drugDiseaseSet = drugDiseaseDMDB.sample(50)

# For each row, build a list: 1 correct drug + 4 distractors
drugDiseaseSet['candidate_drugs'] = drugDiseaseSet.apply(lambda row: get_distractors(drugDiseaseDMDB, row), axis=1)

drugDiseaseSet['question'] = "Which of these drugs can treat " + drugDiseaseSet['disease_name'] + " by targeting " + drugDiseaseSet['bp_name'] + "?\n- " + drugDiseaseSet['candidate_drugs']

drugDiseaseSet.to_excel('Prototype/logs/25questions_07.15.25.xlsx')
