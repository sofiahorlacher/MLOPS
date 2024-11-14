# MLOPS Project

This repository contains a containerized machine learning pipeline. The project is designed to run a training script inside a Docker container, with the ability to track experiments using Weights and Biases (W&B).
Small disclamer: I'm using a Mac so you might need to adjust some things for your OS.

## Preconditions

Before running the project, make sure you have the following installed:

- [Docker Desktop](https://www.docker.com/products/docker-desktop) application
- A terminal/command-line interface
- [Docker Extension](https://marketplace.visualstudio.com/items?itemName=ms-azuretools.vscode-docker) in Visual Studio by Microsoft

## Steps to Run with Docker Locally

### 1. Clone the Repository

First, clone the repository to your local machine:

```bash
git clone https://github.com/sofiahorlacher/MLOPS.git
```

Swith into the project directory:
```bash
cd MLOPS
```

### 2. Build the Docker Image
Create a new Docker context:
```bash
docker context create mlops
docker context use mlops
```
If needed, you can overwrite Docker settings with the --docker argument. 

Then, build the Docker image using the following command:
```bash
docker --context mlops build -t main .
```

### 3. Run the Docker Container
Once the image is built, you can run the training script inside the container with this command:
```bash
docker --context mlops run -it main
```
This run command will use the best hyperparameters for the training. If you want to  run a custom training command you can specify the hyperparameter like this:

```bash
docker --context mlops run -it main --batch_size 64 --learning_rate 2e-04
```

Here is an overview of all the possible parameters to be passed and their default values:

| Argument           | Type    | Default                  | Description                                                                                     |
|--------------------|---------|--------------------------|-------------------------------------------------------------------------------------------------|
| `--model_name_or_path` | `str`   | `"distilbert-base-uncased"` | Pretrained model name or path to be used for training (e.g., `distilbert-base-uncased`).       |
| `--task_name`      | `str`   | `"mrpc"`                 | GLUE task name (e.g., `mrpc`, `sst2`, etc.).                                                   |
| `--max_seq_length` | `int`   | `128`                    | Maximum sequence length for input data.                                                         |
| `--batch_size`     | `int`   | `32`                     | Batch size for training and evaluation.                                                         |
| `--learning_rate`  | `float` | `2e-5`                   | Learning rate for the optimizer.                                                                |
| `--warmup_steps`   | `int`   | `0`                      | Number of warmup steps for learning rate scheduler.                                             |
| `--weight_decay`   | `float` | `0.0`                    | Weight decay (L2 regularization) rate.                                                          |
| `--optimizer`      | `str`   | `"adamw"`                | Optimizer to use (choices: `"adam"`, `"adamw"`).                                                |
| `--scheduler`      | `str`   | `"linear_warmup"`        | Learning rate scheduler to use (choices: `"linear_warmup"`, `"cosine"`).                        |
| `--projectname`    | `str`   | `"test"`                 | Name of the project for logging in Weights & Biases (WandB).                                    |
| `--checkpoint_dir` | `str`   | `"checkpoints"`          | Directory path to save model checkpoints.                                                       |
| `--epochs`         | `int`   | `3`                      | Number of training epochs.                                                                      |




You will be prompted in the terminal to choose how you want to log in to Weights and Biases (W&B). Choose the option that fits you best and follow the steps provided in the console. Should be looking like this:

![Weights & Biases login](images/wandb.png)

If you have an API key from Weights and Biases you can pass it directly into the run command like this:
```bash
docker --context mlops run --env WANDB_API_KEY=key -it main
```

## Troubleshooting
If you encounter any issues:
- Make sure Docker Desktop is running.
- Verify that you are using the correct context by running docker context ls and checking that the context is set to mlops.
- If you have trouble with Docker permissions, try restarting Docker or your terminal.


## Steps to Run only Python Script Locally

### 1. Clone the Repository

First, clone the repository to your local machine:

```bash
git clone https://github.com/sofiahorlacher/MLOPS.git
```

Swith into the project directory:
```bash
cd MLOPS
```

### 2. Create Virtual Environment and Install pip Dependancies

```bash
python3 -m venv myenv
source myenv/bin/activate
python3 -m pip install -r requirements.txt
```

### 2. Run Training Locally

To use the best hyperparameter for the training the command looks like this:

```bash
python src/main.py --checkpoint_dir models --projectname test 
```
If you want to run a custom training command you can specify the hyperparameter like this:

```bash
python src/main.py --checkpoint_dir models --learning_rate 2e-05 --batch_size 32 --optimizer adam --warmup_steps 200 --scheduler linear_warmup --weight_decay 0.0 --projectname test 
```
Check the table above what parameters you can pass.