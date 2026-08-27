# Single image shared by the pipeline, backend and frontend services. They differ
# only by the command they run, so one build serves all three (smaller, simpler).
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# System deps for pyproj/shapely/folium geospatial stack.
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./
RUN pip install --upgrade pip && pip install -r requirements.txt

# Project source
COPY pyproject.toml ./
COPY conf ./conf
COPY src ./src
COPY app ./app
COPY data ./data
COPY docker ./docker

RUN pip install -e . && chmod +x docker/*.sh

EXPOSE 8000 8501
