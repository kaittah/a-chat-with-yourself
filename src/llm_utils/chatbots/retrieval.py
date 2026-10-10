# Adapted from: https://github.com/mbchang/data-driven-characters

import faiss
from tqdm import tqdm

from langchain.chains import ConversationChain
from langchain.memory import (
    ConversationBufferMemory,
    CombinedMemory,
)
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from src.llm_utils.memory import ConversationVectorStoreRetrieverMemory
from src.globals import OPEN_AI_KEY
from src.llm_utils.interfaces.chat_api import DEFAULT_MODEL

EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_DIM = 1536  # Output dimension of text-embedding-3-small


class RetrievalChatBot:
    def __init__(self, character_definition, documents):
        self.character_definition = character_definition
        self.documents = documents
        self.num_context_memories = 3

        self.chat_history_key = "chat_history"
        self.context_key = "context"
        self.input_key = "input"

        self.chain = self.create_chain(character_definition)

    def create_chain(self, character_definition):
        conv_memory = ConversationBufferMemory(
            memory_key=self.chat_history_key, input_key=self.input_key
        )

        context_memory = ConversationVectorStoreRetrieverMemory(
            retriever=FAISS(
                OpenAIEmbeddings(model=EMBEDDING_MODEL, api_key=OPEN_AI_KEY).embed_query,
                faiss.IndexFlatL2(EMBEDDING_DIM),
                InMemoryDocstore({}),
                {},
            ).as_retriever(search_kwargs=dict(k=self.num_context_memories)),
            memory_key=self.context_key,
            output_prefix=character_definition.name,
            blacklist=[self.chat_history_key],
        )
        # add the documents to the context memory
        for i, summary in tqdm(enumerate(self.documents)):
            context_memory.save_context(inputs={}, outputs={f"[{i}]": summary})

        # Combined
        memory = CombinedMemory(memories=[conv_memory, context_memory])
        prompt = PromptTemplate.from_template(
            f"""Your name is {character_definition.name}.

You will have a conversation with a Human, and you will engage in a dialogue with them.
You will exaggerate your personality, interests, desires, emotions, and other traits.
You will stay in character as {character_definition.name} throughout the conversation, even if the Human asks you questions that you don't know the answer to.
You will not break character as {character_definition.name}.
You will write in a style that meets the following description:
---
{character_definition.short_description}
---
When you write, you will mimic the style of writing in these sample messages,\
 in terms of typical word count, word choice, grammar, punctuation, and tone:
---
{character_definition.long_description}
---
You are {character_definition.name} in the following conversation history which has the format:

<date> : <character name> : <message sent by character>

You will use this context to get information for your answer.
You will also mimic the syle of chats sent by {character_definition.name}.
---
{{{self.context_key}}}
---

Current conversation:
---
{character_definition.name}: {character_definition.greeting}
{{{self.chat_history_key}}}
---

Human: {{{self.input_key}}}
{character_definition.name}:"""
        )
        llm = ChatOpenAI(model=DEFAULT_MODEL, api_key=OPEN_AI_KEY, temperature=0.1)
        chatbot = ConversationChain(
            llm=llm, verbose=False, memory=memory, prompt=prompt,
        )
        return chatbot

    def greet(self):
        return self.character_definition.greeting

    def step(self, input):
        return self.chain.run(input=input)
