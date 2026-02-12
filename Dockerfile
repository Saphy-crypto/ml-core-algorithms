# Dockerfile
FROM python:3.12-slim

WORKDIR /app

#install dependencies first 
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

#copy code
COPY . .


CMD ["python", "main.py"]
