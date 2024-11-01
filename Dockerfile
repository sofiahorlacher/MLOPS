# Use an official Python runtime as a parent image
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

COPY requirements.txt /app/

RUN pip install --no-cache-dir -r requirements.txt

COPY . /app

ENV LEARNING_RATE=2e-05
ENV BATCH_SIZE=32
ENV OPTIMIZER=adam
ENV WARMUP_STEPS=200
ENV SCHEDULER=linear_warmup
ENV WEIGHT_DECAY=0.0
ENV PROJECTNAME=test
ENV EPOCHS=3

CMD ["python", "main.py", "--checkpoint_dir", "models", "--learning_rate", "${LEARNING_RATE}", "--batch_size", "${BATCH_SIZE}", "--optimizer", "${OPTIMIZER}", "--warmup_steps", "${WARMUP_STEPS}", "--scheduler", "${SCHEDULER}", "--weight_decay", "${WEIGHT_DECAY}", "--projectname", "${PROJECTNAME}", "--epochs", "${EPOCHS}"]