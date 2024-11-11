# Base image
FROM python:3.10.12-slim

# Set working directory
WORKDIR /app

# Copy the entire project structure
COPY . /app/

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose the port the app runs on
EXPOSE 8000

# Set the environment variable for the config path
ENV CONFIG_PATH=/app/config/default.yaml

# Run the code
CMD ["python3", "src/api/app.py"]