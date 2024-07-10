FROM python:3.11-slim-bookworm

ENV PYTHONUNBUFFERED=1

WORKDIR /api

COPY . /api

RUN apt-get update && apt-get install ffmpeg libsm6 libxext6 poppler-utils tesseract-ocr -y

RUN pip install -r requirements.txt

RUN python -c 'import nltk; nltk.download("punkt"); nltk.download("averaged_perceptron_tagger");'

CMD uvicorn app.main:api --port=8080 --host=0.0.0.0