FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV MPLBACKEND=Agg
ENV OMP_NUM_THREADS=1

WORKDIR /opt/nbodydp

RUN apt-get update && apt-get install -y --no-install-recommends build-essential git && rm -rf /var/lib/apt/lists/*

COPY requirements.txt requirements-dev.txt ./
RUN pip install --no-cache-dir -r requirements-dev.txt

COPY . .
RUN pip install --no-cache-dir -e .

CMD ["python", "scripts/run_full_pipeline.py", "--output", "/opt/nbodydp/results"]
