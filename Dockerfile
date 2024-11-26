# Base image
FROM python:3.10.12-slim

# Set working directory
WORKDIR /app

# Copy folders
# Use the full path relative to where you run docker build
COPY Cameroon-Air-Quality-Prediction/config/ ./config/
COPY Cameroon-Air-Quality-Prediction/src/data/ ./src/data/

# Copy requirements.txt
COPY Cameroon-Air-Quality-Prediction/requirements.txt /app/

# Install dependencies
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt
RUN prefect version

# Expose the port the app runs on
EXPOSE 8080

# Set the environment variable for the config path
ENV CONFIG_PATH=/app/config/default.yaml

# Run the code
CMD ["python3", "src/data/data_pipeline.py"]