FROM python:3.11-slim

WORKDIR /app

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Copy project files
COPY pyproject.toml uv.lock ./

# Install dependencies
RUN uv sync --frozen

# Copy application source
COPY src/ src/
COPY tests/ tests/

# Set Python path
ENV PYTHONPATH=/app

# Default command - run tests
CMD ["uv", "run", "pytest", "tests/", "-v"]