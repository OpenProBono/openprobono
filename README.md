This is our work-in-progress legal AI search tool! This is only the backend/API for the product, we will soon have a frontend where you can create custom bots/tools and try them out yourself!

We use external resources that you will need to set up if you want to recreate the project.
# List of External Resources:
- OpenAI: llm calls
- Serpapi: web search
- Langfuse: tracing
- Milvus/Zilliz: vector database
- Firebase: database
- More that are optional

# Installing & Running
- we recommend using a conda environment, python 3.11, and brew if on mac
- `pip install -r requirements.txt`
- need to set some environment variables for API keys/etc
    - openai, serpapi, milvus/zilliz, firebase.
- `sudo apt-get update && apt-get install ffmpeg libsm6 libxext6 poppler-utils -y` or `brew update && brew install ffmpeg poppler`
- `uvicorn app.main:api --port=8080 --host=0.0.0.0`

# API/Code Documentation
- /docs or main.py has the majority of the documentation, but it is a work in progress. Feel free to reach out with any questions

We welcome discussion and improvements from the community! Engage with us through the discussions & issues on the repo and feel free to email us at arman@openprobono.com
