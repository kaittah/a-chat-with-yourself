import re
import time
import warnings

import pandas as pd
from pandas.errors import SettingWithCopyWarning
import scrubadub, scrubadub_spacy

from src.llm_utils.interfaces.chat_api import get_completion_from_messages, read_string_to_list

warnings.filterwarnings("ignore", category=SettingWithCopyWarning)

def save_prompt_to_file(file_name: str, text: str) -> None:
    with open(f'src/store/prompts/{file_name}.txt', 'w', encoding='utf8') as writer:
        writer.write(text)

def save_response_to_file(file_name: str, text) -> None:
    with open(f'src/store/responses/{file_name}.txt', 'w', encoding='utf8') as writer:
        if type(text) == list:
            for t in text:
                writer.write(str(t))
                writer.write('\n')
        else:
            writer.write(str(text))

def scrub_prompt(prompt: str) -> str:
    scrubber = scrubadub.Scrubber(post_processor_list=[scrubadub.post_processors.FilthReplacer(include_count=True)])
    scrubber.add_detector(scrubadub_spacy.detectors.SpacyEntityDetector)
    clean_text = scrubber.clean(prompt)
    return clean_text

def get_analysis_prompt_pii(grouped_year_data: pd.DataFrame) -> str:
    '''
    Get a batch of text messages to summarize.
    '''
    prompt_data = f"["
    grouped_year_data.sort_values(by='importance_rank', inplace=True)
    for index, row in grouped_year_data.iterrows():
        json_str = "{ " + f"'id': {index}, 'messages': {row['messages']}" + "}" 
        prompt_data = prompt_data + json_str + ", "
    prompt_data = prompt_data + f"]"
    save_prompt_to_file(f'analysis_prompt_{time.time()}', prompt_data)
    return prompt_data

def get_analysis(grouped_year_data: pd.DataFrame) -> list:
    system_message = {'role': 'system', 'content': """
You will receive a python list of objects with the format:
'id': <unique identifier>
'messages': <list of text messages that are separated by pipe characters>                 
Output a list of new python objects, each corresponding to an object in the input. The output should
be a list of objects with the format:
    'id': <the same as the id field in the input>
    'analysis': <text following the below guidlines>
The "analysis" field should only describe the tone, style, grammar, spelling, and language of text messages for that object,
and what you can infer about the author's character, but nothing else.
Describe the manner of communication and do not summarize the content in the analysis. Use less than 20 words.
"""}
    all_responses = []
    for year in grouped_year_data['year'].unique():
        year_data = grouped_year_data.loc[grouped_year_data['year']==year]
        prompt_pii = get_analysis_prompt_pii(year_data)
        prompt = scrub_prompt(prompt_pii)
        user_message = {'role': 'user', 'content': prompt}
        messages = [system_message, user_message]
        chat_response = get_completion_from_messages(messages)
        list_response = read_string_to_list(chat_response)
        all_responses.extend(list_response)
    return all_responses

def get_yearly_change_prompt(grouped_year_data: pd.DataFrame, previous_result: list):
    '''
    Maps each id in the previous list to the corresponding year, sender name, and level of importance
    Creates an object with structure:
       {'2023': []} 
    With years as keys and compiled analyses as values
    '''
    compiled_years = {}
    for object in previous_result:
        importance = grouped_year_data.loc[object['id']]['importance_rank']
        year = grouped_year_data.loc[object['id']]['year']
        analysis = object['analysis']
        compiled_years[year] = compiled_years.get(year, [])
        compiled_years[year].append((analysis, importance))
    for key, analyses in compiled_years.items():
        # Sort the descriptions by importance
        sorted_by_importance = sorted(analyses, key=lambda x: x[1])
        joined_string = '|'.join([s[0] for s in sorted_by_importance])
        compiled_years[key] = joined_string
    return compiled_years

def get_yearly_change_summary(prompt: str):
    system_message = {'role': 'system', 'content': """
You will receive a python object with the structure: <year> : < pipe separated string>. The pipe separated string
represents a sequence of descriptions, with more important descriptions appearing earlier than less important ones.
Use the descriptions to characterize the messages sent in each year. 
                      
Write a 500 word essay-style analysis that describes patterns in messages sent in each chronological year, and what has changed
from year to year. Do not talk about the content of the text messages, only talk about the writing style.
"""}
    save_prompt_to_file(f'yearly_prompt_{time.time()}', prompt)
    user_message = {'role':'user', 'content': prompt}
    messages = [system_message, user_message]
    chat_response = get_completion_from_messages(messages)
    return chat_response

def get_friend_comparison(friend_messages: pd.DataFrame, your_name: str) -> tuple:
    system_message = {'role': 'system', 'content': """
In the section contained in #### You will receive a python list of objects with the format:
'id': <unique identifier>
'messages': <list of text messages that are separated by pipe characters>
In the section contained in $$$$ You will receive a target.
              
For each object, analyze the tone, style, grammar, spelling, and language of text messages for that object,
and what you can infer about the author's character, but nothing else. 
Output the id of the object from the list with messages that are most similar in style to the target. Do not consider
the topics of conversation or the content. Only compare the writing style.
Also output an explanation of how you can tell that this object has the most similar messages based on style, not content.
The output should be in the format:
    'id' : <output id>
    'explanation': <paragraph explanation>
"""}
    if friend_messages.shape[0] > 10:
        friend_messages = friend_messages.sort_values(by='count', ascending=False).iloc[:10]
    prompt_pii = '####\n['
    for index, row in friend_messages.iterrows():
        sender_name = row['sender_name']
        messages = row['messages']
        if sender_name != your_name:
            prompt_pii = prompt_pii + f"""{{'id': {index}, 'messages': {messages}}},"""
    prompt_pii = prompt_pii + ']\n####\n'
    target = friend_messages.loc[friend_messages['sender_name']==your_name]['messages'].values[0]
    prompt_pii = prompt_pii + f"\n$$$$\n{target}\n$$$$"
    prompt = scrub_prompt(prompt_pii)
    save_prompt_to_file(f'friends_{time.time()}', prompt)
    user_message = {'role': 'user', 'content': prompt}
    messages = [system_message, user_message]
    chat_response = get_completion_from_messages(messages)
    try:
        id_field = re.search(""".*['"]id['"].*[0-9]+""", chat_response).group().split(':')[1].strip()
        analysis_field = chat_response[chat_response.index("'explanation':")+15:]
        sender_name = friend_messages.loc[int(id_field)]['sender_name']
        return sender_name, analysis_field
    except:
        return (-1, "Error in response")