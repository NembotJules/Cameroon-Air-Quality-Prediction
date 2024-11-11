# Docker file for the API service...

#first step is to choose a base image

FROM python:3.10.12-slim

# second step is to set the working directory

WORKDIR /app

# third step is to COPY relevant files

COPY src/api  /app
COPY requirements.txt /app

# step 4 is to install dependencies

RUN pip install --no-cache-dir -r requirements.txt

# step 5 is to expose the port the app runs on

EXPOSE 8000

# Run the code

CMD [ "python3", "app.py" ]


