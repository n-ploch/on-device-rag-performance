FROM python:3.13-slim

WORKDIR /app

# Copy shared package first
COPY shared/ /shared/
RUN pip install /shared/

# Copy orchestrator
COPY orchestrator/ /app/
RUN pip install -e .

# Default command
CMD ["python", "-m", "orchestrator.runner"]
