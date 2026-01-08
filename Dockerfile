FROM python:3.10

WORKDIR /app

# Copy backend requirements
COPY SceneForge_Backend/requirements-prod.txt .

# Install Python dependencies
ENV PIP_DEFAULT_TIMEOUT=180
RUN pip install --no-cache-dir -r requirements-prod.txt

# Copy the entire backend code
COPY SceneForge_Backend/app ./app

# Expose port
EXPOSE 8000

# Start the application
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
