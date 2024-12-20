FROM python:3.11-slim-bookworm

ENV PYTHONUNBUFFERED=1

WORKDIR /api

RUN apt-get update && apt-get install ffmpeg git libsm6 libmagic1 libxext6 poppler-utils tesseract-ocr -y

COPY requirements.txt /api/requirements.txt

RUN pip install -r requirements.txt --no-cache-dir

RUN python -m nltk.downloader -d /usr/local/share/nltk_data punkt punkt_tab averaged_perceptron_tagger averaged_perceptron_tagger_eng

COPY app/ /api/app

COPY .git /api/.git

CMD uvicorn app.main:api --port=8080 --host=0.0.0.0