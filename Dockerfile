FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy project files
COPY . .

# Upgrade pip
RUN pip install --upgrade pip

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose port
EXPOSE 10000

# Start server
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "10000"]
