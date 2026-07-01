FROM python:3.10-slim

# Set up a new user 'user' with UID 1000 (required for Hugging Face Spaces)
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH

WORKDIR $HOME/app

# Copy requirements and install (using CPU PyTorch build to keep build time fast)
COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir --upgrade -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu

# Copy the rest of the application files
COPY --chown=user main.py .
COPY --chown=user agent/ ./agent/
COPY --chown=user data/ ./data/

# Expose port 7860 (Hugging Face default)
EXPOSE 7860

# Start FastAPI application using uvicorn on port 7860
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]
