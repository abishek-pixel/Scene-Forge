FROM python:3.10-slim

WORKDIR /app
# Copy both the original requirements and a Docker-friendly copy (without pip hash checks).
# We'll install from the hash-free file to avoid build failures when upstream wheels change.
COPY requirements.txt requirements_docker.txt /app/
# Install requirements using the Docker-friendly file. For production you should pin hashes
# and regenerate them; this is a pragmatic local/dev build approach.
ENV PIP_DEFAULT_TIMEOUT=120
RUN pip --retries 5 install --no-cache-dir -r /app/requirements_docker.txt -f https://download.pytorch.org/whl/cpu/torch_stable.html

# Copy the entire repository into the image so the top-level `app/` package
# is available at runtime (so `uvicorn app.main:app` can import it).
COPY . /app

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]