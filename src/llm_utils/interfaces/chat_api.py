import logging
import re
import time

import openai
from openai.error import RateLimitError, ServiceUnavailableError

from src.globals import OPEN_AI_KEY

openai.api_key=OPEN_AI_KEY

def get_completion_from_messages(messages, 
                                 model="gpt-3.5-turbo", 
                                 temperature=0, 
                                 max_tokens=2000) -> str:
    try:
        response = openai.ChatCompletion.create(
            model=model,
            messages=messages,
            temperature=temperature, 
            max_tokens=max_tokens, 
        )
    except RateLimitError:
        time.sleep(21)
        response = openai.ChatCompletion.create(
            model=model,
            messages=messages,
            temperature=temperature, 
            max_tokens=max_tokens, 
        )
    except ServiceUnavailableError:
        time.sleep(30)
        logging.info('Service unavailable...waiting...')
        response = openai.ChatCompletion.create(
            model=model,
            messages=messages,
            temperature=temperature, 
            max_tokens=max_tokens, 
        )
    return response.choices[0].message["content"]

def read_string_to_list(input_string) -> list:
    result = []
    matches = re.findall(r"""{\s*['"]id['"].*?}""", input_string)
    for m in matches:
        try:
            id_field = re.search(r"""{\s*['"]id['"]:\s*[0-9]+""", m).group().split(':')[1].strip()
            analysis_field = m[m.index("'analysis': '")+12:-2]
            result.append({'id': int(id_field), 'analysis': analysis_field})
        except:
            logging.info(f'Invalid data: {m}')
    return result