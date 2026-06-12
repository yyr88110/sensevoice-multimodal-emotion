FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements first (for Docker cache)
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download models (optional, makes first run faster)
# Uncomment if you want models baked into the image
# RUN python -c "import whisper; whisper.load_model('small')"
# RUN python -c "from funasr import AutoModel; AutoModel(model='iic/SenseVoiceSmall', device='cpu', disable_update=True)"

# Copy source code
COPY . .

# Install the package
RUN pip install --no-cache-dir -e .

# Default entrypoint
ENTRYPOINT ["sensevoice-emotion"]

# Default to showing help
CMD ["--help"]
