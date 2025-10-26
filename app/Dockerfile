# Use a Python base image
FROM python:3.9-slim-buster

# Set the working directory in the container
WORKDIR /app

# Copy only the necessary files for the API first
COPY app/requirements.txt .
COPY .env .env

# Install dependencies for the API, including peft
RUN pip install --no-cache-dir -r requirements.txt

# Copy the LoRA adapters (which includes the tokenizer)
# Assuming models/fine_tuned_t5_lora is parallel to the app/ directory
COPY models/fine_tuned_t5_lora /app/models/fine_tuned_t5_lora

# Copy the application code
COPY app/main.py .

# Expose the port FastAPI will run on
EXPOSE 8000

# Command to run the application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]