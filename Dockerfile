# Docker version 27.2.0
FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime

WORKDIR /app

COPY requirements.txt requirements.txt

RUN pip install -r requirements.txt && pip list --format=freeze > requirements.txt

COPY src src

CMD [ "python", "src/main.py"]
