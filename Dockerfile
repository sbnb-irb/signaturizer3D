FROM python:3.10

WORKDIR /app

COPY pyproject.toml poetry.lock* /app/

RUN pip install poetry

RUN poetry config virtualenvs.create false

RUN poetry install --no-root

RUN poetry run python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
