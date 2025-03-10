FROM python:3.11-slim

# Set the working directory inside the container
WORKDIR /app

# Copy the requirements file and install dependencies
COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire project into the container
COPY . /app/

# Ensure Python includes the /app directory in the Python path
ENV PYTHONPATH="/app:${PYTHONPATH}"

# Expose port 8000 for the FastAPI application
EXPOSE 8000

# Set the entry point to run the FastAPI app with Uvicorn
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]
