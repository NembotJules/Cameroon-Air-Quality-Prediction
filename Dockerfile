# Base image
FROM python:3.10.12-slim

# Set working directory
WORKDIR /app


#Copy folders
COPY config/ ./config/
COPY src/data/ ./src/data/

#Copy requirements.txt
COPY requirements.txt /app/

# Install dependencies
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt
# Install Prefect
RUN pip install prefect==3.1.2

# Ensure Prefect is in the PATH
ENV PATH="/usr/local/bin:${PATH}"

# Expose the port the app runs on
#EXPOSE 8080

# Set the environment variable for the config path
ENV CONFIG_PATH=/app/config/default.yaml

# Run the code
CMD ["python3", "src/data/data_pipeline.py"]