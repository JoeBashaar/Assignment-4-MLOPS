FROM python:3.10-slim

ARG RUN_ID

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Simulate downloading the model from MLflow
RUN echo "Downloading model for RUN_ID=${RUN_ID}"

CMD ["echo", "Model server ready."]
