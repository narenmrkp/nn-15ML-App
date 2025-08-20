# ===========================
# Dockerfile for 15-ML Streamlit App
# ===========================

# Base image with Python 3.12
FROM python:3.12-slim

# Set working directory
WORKDIR /app

# Copy local project files to container
COPY . /app

# Upgrade pip
RUN python -m pip install --upgrade pip

# Install required libraries
RUN pip install --no-cache-dir -r requirements.txt

# ===========================
# IMPORTANT: Do NOT hardcode your HF API key here!
# Instead, pass it at runtime using -e HF_API_KEY=your_hf_api_key
# Example:
#   docker run -p 8501:8501 -e HF_API_KEY=<your_token> ml_portfolio:latest
# ===========================

# Expose Streamlit default port
EXPOSE 8501

# Set environment variables (optional defaults; override at runtime)
# ENV HF_API_KEY=your_hf_api_key    <-- DO NOT uncomment, for security only!

# Streamlit configuration (avoid asking for user input inside container)
ENV STREAMLIT_SERVER_ENABLECORS=false
ENV STREAMLIT_SERVER_HEADLESS=true
ENV STREAMLIT_SERVER_PORT=8501

# Run the main app
CMD ["streamlit", "run", "main_app.py", "--server.port=8501", "--server.headless=true"]
