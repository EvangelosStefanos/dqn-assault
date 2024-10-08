# Docker version 27.2.0
FROM python:3.9

WORKDIR /app

COPY requirements.txt requirements.txt

RUN pip install -r requirements.txt

COPY src src

CMD [ "python", "src/main.py"]
