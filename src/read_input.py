from zipfile import ZipFile
from collections import Counter
import json
import emoji
import pandas as pd
from ftfy import fix_text
import re


def clean_text(text):
    if pd.isna(text):
        return None
    text = fix_text(text)
    text = emoji.replace_emoji(text)
    text = text.strip()
    text = text.replace('\n', ' ')
    return text

def extract_messages(input_file_path) -> pd.DataFrame:
    '''
    Given zip file in format that Facebook provides it, collect information about each message
    that you sent or received and return a dataframe.

    {
        sent_at: [timestamp]
        sent_to: [list]
        message_text: [string]
        is_reaction: [bool]
        clean_text: [string]
    }
    '''

    corpa = []

    with ZipFile(input_file_path, 'r') as z:
        for filename in z.namelist():
            if re.match('.*message[_0-9]+\.json', filename):
                with z.open(filename) as f:
                    data = json.load(f)
                    corpa.append(data)
    names_counter = Counter()
    for c in corpa:
        p = [a['name'] for a in c['participants']]
        names_counter.update(p)

    # edge case - you have very few conversations
    your_name = names_counter.most_common(1)[0][0]

    message_to_people_combined = []
    for c in corpa:
        p = [a['name'] for a in c['participants']]
        messages = c['messages']
        messages_plus_reactions = []
        to_people = list(filter(lambda x: x != your_name, p))


        for m in messages:
            if 'timestamp_ms' in m.keys():
                m['sent_at'] = m.pop('timestamp_ms')
            m['sent_to'] = ','.join(to_people)[:30]
            m['n_recipients'] = len(to_people)
            messages_plus_reactions.append({
                'sender_name': m['sender_name'],
                'sent_at': m['sent_at'],
                'sent_to': m['sent_to'],
                'n_recipients': m['n_recipients'],
                'content': m.get('content', ''),
                'is_reaction': m.get('content','')[:9] == 'Reacted â'
                }
            )
            # Consider a reaction another message
            for r in m.get('reactions', []):
                messages_plus_reactions.append({
                    'sender_name': r['actor'],
                    'sent_at': m['sent_at'],  # Reactions happen after the message, but use message sent_at as a placeholder
                    'sent_to': m['sent_to'],
                    'n_recipients': m['n_recipients'],
                    'content': 'Reacted ' + r['reaction'],
                    'is_reaction': True 
                })

        message_to_people_combined.extend(messages_plus_reactions)
    messages_df = pd.DataFrame(message_to_people_combined)
    messages_df['sent_at'] = pd.to_datetime(messages_df['sent_at'], unit='ms')
    messages_df['clean_text'] = messages_df['content'].apply(clean_text)
    messages_df['date'] = messages_df['sent_at'].dt.date
    messages_df['month'] = messages_df['date'] + pd.offsets.MonthBegin(-1)
    messages_df['year'] = messages_df['sent_at'].dt.year
    messages_df.dropna(inplace=True)
    your_messages_df = messages_df.loc[messages_df['sender_name']==your_name]
    one_on_one_messages_df = messages_df.loc[messages_df['n_recipients']==1]
    return your_messages_df, one_on_one_messages_df