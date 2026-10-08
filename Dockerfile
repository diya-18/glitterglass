FROM python:3.12-slim
ENV PYTHONUNBUFFERED=1
RUN apt-get update && apt-get install -y --no-install-recommends gcc libc6-dev && rm -rf /var/lib/apt/lists/*
WORKDIR /app
RUN pip install --no-cache-dir numpy nltk
COPY . .
RUN gcc -O3 -shared -fPIC -o searchengine/kernel.so searchengine/kernel.c
CMD ["python", "-m", "searchengine.server", "indexes/mini", "data/mini.tsv"]
