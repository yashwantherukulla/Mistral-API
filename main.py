import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, pipeline

import os

from langchain.llms import HuggingFacePipeline
from langchain.embeddings import HuggingFaceEmbeddings
from langchain import HuggingFacePipeline

from fastapi import Request,FastAPI, Body
import nest_asyncio
import uvicorn

from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

import sys
from huggingface_hub import login

import threading


login(token=sys.argv[1])

model_name = "mistralai/Mistral-7B-Instruct-v0.2"

def load_quantized_model(model_name: str):
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        quantization_config=bnb_config
    )
    return model

model = load_quantized_model(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")
tokenizer.pad_token = tokenizer.eos_token

from langchain.llms import HuggingFacePipeline

pipeline = pipeline (
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    use_cache=True,
    device_map="auto",
    max_length=32000,
    do_sample=True,
    top_k=5,
    num_return_sequences=1,
    eos_token_id=tokenizer.eos_token_id,
    pad_token_id=tokenizer.pad_token_id,
)


llm = HuggingFacePipeline(pipeline=pipeline)



prompt = PromptTemplate(
    template="""<s>[INST] 
    You are a helpful and knowledgeable chatbot. You should provide accurate, clear, and concise responses to user questions based on the provided context and conversation history. 

Inputs:

Question: The user's current question.
History: The past 15 interactions with the user, formatted as alternating "User:" and "Assistant:" exchanges. Each exchange includes both the user’s input and the chatbot’s response for the respective user input.
Desired Response:

- Begin by acknowledging the context from the conversation history if relevant.
- Answer the user's current question accurately and concisely.
- If the question is unclear, ask for clarification.
- Maintain a friendly and professional tone.
- Only output your response. Do not output the entire message that is given to you.
- Make sure that your responses are not repeated from your chat history.

  USER QUESTION:\n
  {question}\n\n
  CHAT HISTORY:\n
  {history}\n\n
     [/INST]</s>""",
    input_variables=["question", "history"],
)

chain = prompt | llm | StrOutputParser()


chat_history = []

def conv(query:str, chat_history:list):
  history = str(chat_history[-15:])
  if len(history)>15000:
    history[:15000]
  response = chain.invoke({"question": query,"history": history})
  response = response.split("[/INST]</s> ")
  response = response[-1]
  chat_history.append({"User" : query, "Assistant" : response})

  return response, chat_history



nest_asyncio.apply()
app = FastAPI()
chat_history = []



# @app.get("/")
# async def root():
#     return {"message": "Hello World"}

@app.post('/endpoint/')
async def conv(request: Request):
  data = await request.json()
  query =data["query"]
  history = str(chat_history[-15:])
  if len(history)>15000:
    history[:15000]
  response = chain.invoke({"question": query,"history": history})
  response = response.split("[/INST]</s> ")
  response = response[-1]
  chat_history.append({"User" : query, "Assistant" : response})

  return {"response" : response}


def run_server():
  uvicorn.run(app, host="127.0.0.1", port=8000, loop="asyncio")
threading.Thread(target=run_server).start()