# Docker version 27.2.0
FROM python:3.9

WORKDIR /app

RUN pip install torch torchvision torchaudio

RUN pip install gymnasium[atari,accept-rom-license] matplotlib torchinfo moviepy

COPY app/src src/

CMD [ "python", "src/main.py"]
