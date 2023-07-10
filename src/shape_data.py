from src.globals import MAX_MESSAGES, MAX_PEOPLE

def get_messages_for_year(year, messages, fraction=1):
    '''
    Returns messages sent during the given year
    '''
    return messages.loc[messages['sent_at'].dt.year==year].sample(frac=fraction)

def get_first_ever_message(messages):
    return messages.sort_values(by='sent_at').iloc[0][['clean_text', 'sent_at', 'sent_to']]

def get_avg_messages(messages):
    '''
    Each month, what is the average number of messages you sent per day
    '''
    month_count = messages.groupby(['month'])[['date']].agg(['count', 'nunique']).reset_index()
    month_count.columns = ['month', 'messages', 'days']
    month_count['messages_per_day'] = month_count['messages']/month_count['days']
    return month_count[['month', 'messages_per_day']]

def shuffle_messages(messages):
    messages_sample = messages.sample(frac=1).dropna()
    # Remove messages that are just reactions, changing something in the chat, or a a few words
    messages_sample = messages_sample.loc[(~messages_sample['is_reaction']) & (~messages_sample['clean_text'].str.find('You')==0) & (messages_sample['clean_text'].str.strip().str.count(' ') > 2)]
    return messages_sample

def group_by_year_and_sender(messages):
    messages_yr = shuffle_messages(messages)
    #Get count of messages and a string combining random messages for each person and year
    messages_grouped = messages_yr.groupby(['year', 'sent_to'])[['clean_text']].agg(['count', lambda x: '|'.join(x[-MAX_MESSAGES:])]).reset_index()
    messages_grouped.columns=['year', 'sent_to', 'n_messages', 'messages']
    messages_grouped['importance_rank'] = messages_grouped.groupby('year')['n_messages'].rank('dense', ascending=False)
    
    return messages_grouped.loc[messages_grouped['importance_rank'] <= MAX_PEOPLE]

def sample_latest_year_by_sender(messages):
    latest_year = messages['year'].max()
    latest_year_messages = messages.loc[messages['year'] == latest_year]
    latest_year_messages = shuffle_messages(latest_year_messages)
    messages_grouped = latest_year_messages.groupby(['sender_name'])[['clean_text']].agg(['count', lambda x: '|'.join(x[-MAX_MESSAGES:])]).reset_index()
    messages_grouped.columns=['sender_name', 'count', 'messages']
    return messages_grouped
