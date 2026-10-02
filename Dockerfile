FROM python:3.11-slim

WORKDIR /app

# gcc/g++ are required to compile faiss-cpu native extensions
RUN apt-get update \
    && apt-get install -y --no-install-recommends gcc g++ \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml .

# Install only production dependencies (no dev extras)
RUN pip install --no-cache-dir -e .

COPY app/ ./app/

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
