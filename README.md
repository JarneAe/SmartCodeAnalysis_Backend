# AI Backend

## Getting started

1. Clone the Repository

Run the following command in your terminal to clone the repository:

```bash
git clone <repository-url>
cd <repository-folder>
```

2. Run the Docker Compose

Ensure Docker is installed and running on your machine. Then, use the following command to build and start the services:

```bash
docker-compose up --build
```

**This will start the required containers:**

- web: The FastAPI-based backend.
- qdrant: The vector database for embeddings.
- ollama: The pre-trained language model for generating responses.


3. Access the API Documentation

Once the services are up, open your browser and navigate to:

http://localhost:8001/docs

This will open the Swagger-UI, where you can:

- View all available API endpoints.
- Test the endpoints directly from the browser.

--- 
#Technical details

## Diagram