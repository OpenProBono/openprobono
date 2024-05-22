FROM python:3.11-slim-bookworm

ENV PYTHONUNBUFFERED=1

WORKDIR /api

COPY . /api

RUN apt-get update && apt-get install ffmpeg libsm6 libxext6 poppler-utils -y

RUN pip install -r requirements.txt

CMD uvicorn main:api --port=8080 --host=0.0.0.0