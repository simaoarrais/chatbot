FROM python:3.10-slim

WORKDIR /app

# Copy and install the application dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy in the source code
COPY . .

# Expose Streamlit port
EXPOSE 8501

ENTRYPOINT ["streamlit", "run", "chatbot.py", "--server.port=8501", "--server.address=0.0.0.0"]
