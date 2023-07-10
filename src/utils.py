import plotext as plt
import pandas as pd
import time
from termcolor import colored
import random
import os

from src.globals import RAW_FILE_PATH
from src.read_input import extract_messages
from src.shape_data import get_first_ever_message, get_avg_messages

def get_random_alias():
    return 'person-' + str(random.choice(range(9999)))

def get_data(data):
    if os.path.exists('src/store/input/your_messages.csv') and os.path.exists('src/store/input/1_1_messages.csv'):
        your_messages = pd.read_csv('src/store/input/your_messages.csv', parse_dates=['sent_at', 'month'])
        one_on_one_messages = pd.read_csv('src/store/input/1_1_messages.csv', parse_dates=['sent_at', 'month'])
    else:
        your_messages, one_on_one_messages = extract_messages(RAW_FILE_PATH)
        your_messages.to_csv('src/store/input/your_messages.csv', index=False)
        one_on_one_messages.to_csv('src/store/input/1_1_messages.csv')
    data['your_messages'] = your_messages
    your_name = your_messages['sender_name'].min()
    first_message, first_message_sent_on, first_message_sent_to =  get_first_ever_message(your_messages)
    data['first_message'] = first_message
    data['first_message_sent_on'] = first_message_sent_on
    data['first_message_sent_to'] = first_message_sent_to
    data['avg_msg_per_day'] = get_avg_messages(your_messages)
    sender_names = one_on_one_messages['sender_name'].unique()
    alias_mapping = {}
    for name in sender_names:
        random_alias = get_random_alias()
        while random_alias in alias_mapping.keys():
            random_alias = get_random_alias()
        alias_mapping[random_alias] = name
    name_to_alias_mapping = {}
    for alias, name in alias_mapping.items():
        name_to_alias_mapping[name] = alias
    your_alias = name_to_alias_mapping[your_name]
    data['your_alias'] = your_alias
    data['your_name'] = your_name
    one_on_one_messages['sender_name'] = one_on_one_messages['sender_name'].replace(name_to_alias_mapping)  # Protect privacy
    data['one_on_one_messages'] = one_on_one_messages
    return data

def type_slow(string, leave_end=False, color=None):
    for char in string:
        if color:
            print(colored(char, color, attrs=["bold"]), end='')
        else:
            print(char, end='')
        time.sleep(.05)
    if not leave_end:
        print('')
        time.sleep(len(string)//25 + .5)
        
def plot_avg_msg(df):
    plt.clear_figure()
    plt.limit_size()
    x_axis = plt.datetimes_to_string([pd.to_datetime(d) for d in df['month'].values])
    y_axis = list(df['messages_per_day'].values)
    plt.bar(x_axis, y_axis)
    plt.show()
    input('Press enter to continue')
