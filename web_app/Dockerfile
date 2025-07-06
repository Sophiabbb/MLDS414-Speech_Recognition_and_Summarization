# Use Python 3.12.2
FROM python:3.12.2-slim

# Set working directory inside container
WORKDIR /app

# Copy project files into the container
COPY . /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
RUN pip install --upgrade pip \
 && pip install -r requirements.txt \
 && python -m spacy download en_core_web_sm

# Set environment variables
ENV PYTHONUNBUFFERED=1

# Run the pipeline
CMD ["streamlit", "run", "webapp.py"]
