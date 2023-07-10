from src.shape_data import (
    group_by_year_and_sender, get_messages_for_year, sample_latest_year_by_sender
)
from src.llm_utils.functions.prompt import (
    get_analysis, save_response_to_file, get_yearly_change_prompt, scrub_prompt, get_yearly_change_summary,
    get_friend_comparison
)
from src.llm_utils.functions.chat import run_chatbot, get_corpus
from src.utils import type_slow, plot_avg_msg, get_data
import plotext as plt
import time
import threading

def main():
    data = {}
    thread_for_data = threading.Thread(target=get_data, args=(data,))
    thread_for_data.start()
    counter=0
    plt.clear_terminal()
    print('Loading', end = ' ')
    while thread_for_data.is_alive():
        print('|',end='')
        time.sleep(.5)
        counter += 1
        if counter == 20:
            print('')
            print('Get ready!', end = ' ')
        elif counter == 37:
            print('')
            print('Almost there', end = ' ')
        elif counter == 52:
            print('')
            print('Promise.....')        
    plt.clear_terminal()
    type_slow("Okay it's ready! Are you ready? ")
    input('Press enter to continue')
    plt.clear_terminal()

    type_slow("Okay...let's see...this was your first ever sent-message on Messenger...")
    type_slow(f"""message: {data['first_message']}\nsent on: {data['first_message_sent_on']}\n
conversation with: {data['first_message_sent_to']}""")
    input('Press enter to continue')
    plt.clear_terminal()
    type_slow('😂🤣'*10)
    plt.clear_terminal()
    type_slow("ChatGPT, can you look at these messages I sent and tell me about them, particularly how they differ year over year?")
    grouped = group_by_year_and_sender(data['your_messages'])
    analysis_result = get_analysis(grouped)
    save_response_to_file(f"analysis_result_{time.time()}", analysis_result)
    year_summaries = get_yearly_change_prompt(grouped, analysis_result)
    save_response_to_file(f"year_summaries_{time.time()}", year_summaries)
    prompt_pii = str(year_summaries)
    prompt = scrub_prompt(prompt_pii)
    year_summary_result = get_yearly_change_summary(prompt)
    save_response_to_file(f"year_summary_result_{time.time()}", year_summary_result)
    plt.clear_terminal()
    print(year_summary_result)
    input('Press enter to continue')
    type_slow("Essays are nice and all, but let's see if we can talk directly with your past self.")
    input('Press enter to continue')
    plt.clear_terminal()
    type_slow("""Now entering chat mode. Submit 'exit' in order to exit a chat conversation.\n Otherwise, enjoy the conversation!"""
    )
    selected_continue = True
    while selected_continue:
        try:
            selected_year = int(input('Input a year you were active on Messenger during: '))
        except:
            selected_year = 'Not an int'
        if selected_year in year_summaries.keys():
            type_slow(f'Entering a chat with you at the end of {selected_year}...')
            corpus = get_corpus(data['one_on_one_messages'], selected_year, frac=.25)
            run_chatbot(year_summaries[selected_year],
                        get_messages_for_year(selected_year, data['your_messages'], fraction = .5),
                        data['your_alias'],
                        corpus
            )
        else:
            type_slow(f'We do not have data on you from that time. Try a different year.')  
        input_continue = input('Would you like to chat with you from another year? (Y/N): ')
        if input_continue.lower().strip() == 'y':
            plt.clear_terminal()
            continue
        else:
            selected_continue = False
    plt.clear_terminal()
    type_slow('Welcome back to the present!')
    type_slow('You might be wondering which of your friends has a writing style most similar to yours. We can check that')
    sampled = sample_latest_year_by_sender(data['one_on_one_messages'])
    sender_name, friend_analysis = get_friend_comparison(sampled, data['your_alias'])
    type_slow('We got an answer from ChatGPT: ')
    save_response_to_file(f'friend_comparison_result_{time.time()}', friend_analysis)
    print(friend_analysis)
    type_slow(f"It's talking about {sender_name}!")
    input('Press enter to continue')
    plt.clear_terminal()
    type_slow("I'll leave you with one more thing to ponder")
    type_slow("Here are your average messages sent per day in each calendar month!")
    plot_avg_msg(data['avg_msg_per_day'])
    plt.clear_terminal()
    type_slow("Bye for now!")

if __name__ == '__main__':
    main()