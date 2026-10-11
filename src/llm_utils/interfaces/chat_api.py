import logging
import re
import time

from openai import OpenAI, RateLimitError, InternalServerError

from src.globals import OPEN_AI_KEY

DEFAULT_MODEL = "gpt-6-luna"

client = OpenAI(api_key=OPEN_AI_KEY)

def get_completion_from_messages(messages, 
                                 model=DEFAULT_MODEL, 
                                 max_completion_tokens=2000) -> str:
    def create():
        return client.chat.completions.create(
            model=model,
            messages=messages,
            max_completion_tokens=max_completion_tokens,
        )
    try:
        response = create()
    except RateLimitError:
        time.sleep(21)
        response = create()
    except InternalServerError:
        logging.info('Service unavailable...waiting...')
        time.sleep(30)
        response = create()
    return response.choices[0].message.content

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