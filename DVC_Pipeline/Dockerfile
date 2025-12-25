FROM python:3.11.0

WORKDIR /water-potability-pred

RUN apt-get update && apt-get install -y --no-install-recommends git \
&& rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN pip install -r requirements.txt


COPY . .


CMD ["sh", "-lc", "git init && dvc pull && dvc repro"]

