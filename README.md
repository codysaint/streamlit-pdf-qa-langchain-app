# streamlit-pdf-qa-langchain-app
This repo contains code to query PDF(s) using langchain, GPT, chromadb, tiktoken, and streamlit

## mydocs directory contains local docs/knowledge resource.

you can clone the repo and replace the docs with yours and try to run the bot.

-------------

For running it locally, perform below steps-

- `git clone https://github.com/codysaint/streamlit-pdf-qa-langchain-app.git`
  
- `cd streamlit-pdf-qa-langchain-app`

- create a new virtual environment called `.venv`
  
  + `python -m venv .venv`
  
- Activate the virtual environment
  
  + `.venv\Scripts\activate`

- Install the project requirements
  
  + `pip install -r requirements.txt`

- Delete the files in `vector_index` directory so as to hold the vectors of your own document.
  
- Start the app using the following command:
  
`streamlit run app.py`
