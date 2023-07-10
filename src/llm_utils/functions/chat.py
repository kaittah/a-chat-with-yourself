from dataclasses import dataclass, asdict
import math
import random

from langchain.text_splitter import RecursiveCharacterTextSplitter

from src.llm_utils.chatbots import RetrievalChatBot
from src.llm_utils.interfaces import CommandLine
from src.llm_utils.functions.prompt import scrub_prompt

import pandas as pd

OUTPUT_ROOT = "output"

@dataclass
class Character:
    name: str
    short_description: str
    long_description: str
    greeting: str

def get_corpus(messages: pd.DataFrame, year: int, frac: float = 1) -> None:
    '''
    Only use chat history from prior 2 years, representing actual memory
    Sample random months to cut down on size
    '''
    random.seed(42)
    months = range(1, 13)
    n_months = math.floor(frac*12)
    random_months = random.choices(months, k=n_months)

    filtered_messages = messages.loc[(messages['year']>year-2) & (messages['year']<=year) & (messages['sent_at'].dt.month.isin(random_months))]
    sorted_messages = filtered_messages.sort_values(by=['sent_to', 'sent_at'])
    sorted_messages = sorted_messages.loc[pd.notna(sorted_messages['sender_name'])]
    sorted_messages = sorted_messages.loc[pd.notna(sorted_messages['sent_to'])]
    combined = ''
    for _, row in sorted_messages.iterrows():
        text = row['clean_text']
        combined = combined + f'{row["date"]} : {row["sender_name"]} : {text}\n'
        if len(combined) > 1e5:
            combined = combined[:100000]
            break
    clean = scrub_prompt(combined)
    return clean

def get_docs(corpus: str) -> list:
    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=256, chunk_overlap=0
    )
    docs = [ x.page_content for x in text_splitter.create_documents([corpus])]
    return docs

def get_character_definition(year_summary: str, messages_df: pd.DataFrame, your_alias: str)-> Character:
    short_description = year_summary
    long_description = ""
    for i in range(1,11):
        if len(long_description) > 500:
            long_description = long_description[:500]
            break
        if i > messages_df.shape[0]:
            break
        long_description = long_description + f"{i}. {messages_df.iloc[i]['clean_text']} \n"
    # get character definition
    character_definition = Character(
        name=your_alias,
        short_description=short_description,
        long_description=long_description,
        greeting='Hi',
    )

    return character_definition

def create_chatbot(year_summary, messages_df, your_alias, corpus):
    chatbot = RetrievalChatBot(
            character_definition=get_character_definition(year_summary, messages_df, your_alias),
            documents=get_docs(corpus),
    )
    return chatbot


def run_chatbot(year_summary, messages_df, your_alias, corpus):
    chatbot = create_chatbot(
        year_summary,
        messages_df,
        your_alias,
        corpus
    )
    app = CommandLine(chatbot=chatbot)
    app.run()
