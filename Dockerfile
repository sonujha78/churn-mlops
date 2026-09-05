FROM python:3.12-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Configure pip for slow/unstable networks
ENV PIP_DEFAULT_TIMEOUT=120
ENV PIP_RETRIES=10

RUN pip install --no-cache-dir --upgrade pip

# Install in stages so Docker can cache each layer separately
RUN pip install --no-cache-dir numpy pandas scikit-learn
RUN pip install --no-cache-dir xgboost
RUN pip install --no-cache-dir mlflow
RUN pip install --no-cache-dir fastapi "uvicorn[standard]" pydantic

# Copy source code
COPY src/ ./src/

# Environment variables (can be overridden at runtime)
ENV MLFLOW_TRACKING_URI="sqlite:///mlflow.db"
ENV MODEL_NAME="churn-model"
ENV MODEL_ALIAS="production"

EXPOSE 8000

CMD ["uvicorn", "src.serve:app", "--host", "0.0.0.0", "--port", "8000"]
