FROM python:3.11-slim-buster

RUN apt-get update                                 \
 && apt-get install -y --reinstall build-essential \
 && apt-get install -y libgomp1 gcc vim git        \
 && apt-get install -y postgresql-server-dev-all

COPY mlflow_docker/requirements.txt  /app/requirements.txt
WORKDIR /app

RUN pip install --no-cache-dir -r requirements.txt

