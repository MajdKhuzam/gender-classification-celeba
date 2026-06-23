# Base image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Copy requirements FIRST (before the rest of the code)
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the project
COPY . .

EXPOSE 7860

# Run the app
CMD ["gunicorn", "main:app", \
     "-k", "uvicorn.workers.UvicornWorker", \
     "--workers", "1", \
     "--bind", "0.0.0.0:7860", \
     "--timeout", "120"]