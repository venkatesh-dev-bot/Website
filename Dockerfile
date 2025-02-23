FROM python:3.10-slim

# Set the working directory in the container
WORKDIR /app

# Copy only requirements and setup files first to take advantage of Docker cache
COPY requirements.txt ./

# Install dependencies and package in editable mode
RUN pip install -r requirements.txt

# Copy .env file (if you have one)
COPY .env ./

# Copy source code into the container
COPY . /app

# Set environment variables
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

# Expose the port the app will run on
EXPOSE 9090

# # Run the Uvicorn application
# CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "9090"]
CMD ["./run.sh"]
