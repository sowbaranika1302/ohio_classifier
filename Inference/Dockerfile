FROM python:3.11-slim
WORKDIR /ohio_classifier
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
ENTRYPOINT ["python", "inference.py"]