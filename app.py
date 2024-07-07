import os
from apikey import apikey
import streamlit as st
from langchain_openai import OpenAI, ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, SimpleSequentialChain
from langchain.globals import set_verbose


os.environ["OPENAI_API_KEY"] = apikey

st.title('Medium Article Generator')

topic = st.text_input('Input your topic of interest')
#language = st.text_input('Language')

title_template = PromptTemplate(input_variables=['topic'],
                                template='Give me a medium article title on {topic}')

article_template = PromptTemplate(input_variables=['title'],
                                  template='Give me a medium article for {title} including the title')

llm = OpenAI(temperature=0.9)
llm2 = ChatOpenAI(model_name='gpt-4', temperature=0.9)

set_verbose(True)
title_chain = LLMChain(llm=llm, prompt=title_template, verbose=True)
article_chain = LLMChain(llm=llm2, prompt=article_template)

overall_chain = SimpleSequentialChain(chains=[title_chain, article_chain], verbose=True)


if topic:
    #response = title_chain.invoke({'topic': topic,'language': language})
    #st.write(response['text'])
    response = overall_chain.invoke(topic)
    st.write(response['output'])
