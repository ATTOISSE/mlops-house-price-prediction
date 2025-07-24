FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt ./
COPY app.py ./
COPY models/ ./models/

RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:8000/loggers')"

CMD ["python", "app.py"]
