This is our work-in-progress legal AI search tool! This is only the backend/API for the product, we will soon have a frontend where you can create custom bots/tools and try them out yourself!

We use external resources that you will need to set up if you want to recreate the project. The minimum requirements are credentials for [OpenAI](https://platform.openai.com/api-keys) or [Anthropic](https://console.anthropic.com/settings/keys), [Milvus](https://milvus.io/docs/install-overview.md), [Firebase](https://firebase.google.com/docs/admin/setup#initialize_the_sdk_in_non-google_environments), [SerpApi](https://serpapi.com/users/sign_up), and [Langfuse](https://langfuse.com/faq/all/where-are-langfuse-api-keys). 

# List of External Resources:
- OpenAI/Anthropic/Other Providers: llm calls
- SerpApi/Google: web search
- Langfuse: tracing
- Zilliz(Milvus): vector database
- Firebase: database
- More that are optional

# Docker Install
You will need [Docker](https://docs.docker.com/get-docker/) and [Docker Compose](https://docs.docker.com/compose/install/) installed on your machine. Clone the repository, add your credentials in `docker-compose.yml`, and enter the following commands:
```
cd openprobono
cp docker-compose.sample.yml docker-compose.yml
docker compose up
```

# Manual Install
- we recommend using a conda environment, python 3.11, and brew if on mac
- `pip install -r requirements.txt`
- need to set some environment variables for API keys/etc
    - openai, serpapi, milvus/zilliz, firebase.
- `sudo apt-get update && apt-get install ffmpeg libsm6 libxext6 poppler-utils -y` or `brew update && brew install ffmpeg poppler`
- `uvicorn app.main:api --port=8080 --host=0.0.0.0`

# API/Code Documentation
- /docs or main.py has the majority of the documentation, but it is a work in progress. Feel free to reach out with any questions

# DB Schema
- Firebase will create the database collections on the fly, no need to set up a schema.
- Milvus/Zilliz, here are our main schemas, we mainly only ever change the dimension of the vector. You can create these collections using our create_collection function as well. You may want to consider using Milvus Lite if trying to run locally.
  - <img width="800" alt="Screenshot 2024-11-14 at 12 05 39 PM" src="https://github.com/user-attachments/assets/1c60e89a-720f-4474-a6f2-fcb8848604cd">
  - <img width="800" alt="Screenshot 2024-11-14 at 12 07 02 PM" src="https://github.com/user-attachments/assets/db5e6f01-b97a-4f31-98fe-467300e6957f">


We welcome discussion and improvements from the community! Engage with us through the discussions & issues on the repo and feel free to email us at arman@openprobono.com
