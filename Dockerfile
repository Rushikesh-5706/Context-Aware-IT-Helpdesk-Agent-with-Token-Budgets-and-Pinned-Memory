FROM python:3.11-slim

WORKDIR /app

# Install dependencies first so Docker layer cache is useful when only code changes
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source
COPY src/ ./src/
COPY eval_5_turns.py .

# logs/ volume is mounted at runtime; create the directory so the app can write
# to it even before the volume is attached (defensive for non-Docker runs).
RUN mkdir -p logs

EXPOSE 8000

CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]
