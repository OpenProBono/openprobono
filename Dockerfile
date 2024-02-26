FROM python:3.11-slim-bookworm

ENV PYTHONUNBUFFERED=1

WORKDIR /api

COPY . /api

RUN pip install -r requirements.txt

CMD uvicorn main:api --port=8080