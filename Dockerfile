# Sets the Python version to 3.12 for the container
FROM python:3.11-slim

# Set the working directory
WORKDIR /code

# Copy the requirements file
COPY ./requirements.txt /code/requirements.txt

# Install the dependencies
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

# Copy the application code
COPY ./app /code/app

ENV WANDB_PROJECT=''
ENV WANDB_MODEL_NAME=''
ENV WANDB_MODEL_VERSION=''
ENV WANDB_API_KEY=''
ENV WANDB_ORG=''

EXPOSE 8080

# Command to run the application
# same as fastapi run app/main.py --port 8080 

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8080"]