# For more information, please refer to https://aka.ms/vscode-docker-python
FROM python:3.8-slim

# Keeps Python from generating .pyc files in the container
ENV PYTHONDONTWRITEBYTECODE=1

# Turns off buffering for easier container logging
ENV PYTHONUNBUFFERED=1

# Install pip requirements
COPY requirements.txt .
RUN python -m pip install -r requirements.txt

WORKDIR /app
COPY . /app

# Creates a non-root user with an explicit UID and adds permission to access the /app folder
# For more info, please refer to https://aka.ms/vscode-docker-python-configure-containers
RUN adduser -u 5678 --disabled-password --gecos "" appuser && chown -R appuser /app
USER appuser

ENV LEARNING_RATE=2e-05
ENV BATCH_SIZE=32
ENV OPTIMIZER=adam
ENV WARMUP_STEPS=200
ENV SCHEDULER=linear_warmup
ENV WEIGHT_DECAY=0.0
ENV PROJECTNAME=test
ENV EPOCHS=3

# During debugging, this entry point will be overridden. For more information, please refer to https://aka.ms/vscode-docker-python-debug
CMD ["python", "main.py", "--checkpoint_dir", "models", "--learning_rate", "${LEARNING_RATE}", "--batch_size", "${BATCH_SIZE}", "--optimizer", "${OPTIMIZER}", "--warmup_steps", "${WARMUP_STEPS}", "--scheduler", "${SCHEDULER}", "--weight_decay", "${WEIGHT_DECAY}", "--projectname", "${PROJECTNAME}", "--epochs", "${EPOCHS}"]