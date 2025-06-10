#!/bin/bash

PROJECT_NAME=mlops \
  MAGE_CODE_PATH=/home/src \
  docker compose -f docker-compose.mlflow.yaml up -d --build
