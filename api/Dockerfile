FROM python:3.12-slim

WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1

COPY pyproject.toml requirements.txt ./
COPY src ./src
# Exact versions and hashes, so an image built next month runs the same code as today's tests.
RUN pip install --no-cache-dir --require-hashes -r requirements.txt && pip install --no-cache-dir --no-deps .
RUN useradd --create-home --uid 1000 app
USER app

EXPOSE 8000
# Schema setup runs separately: `docpilot db init` (compose command / Fly release_command).
CMD ["uvicorn", "docpilot.app:app", "--host", "0.0.0.0", "--port", "8000", "--no-access-log"]
