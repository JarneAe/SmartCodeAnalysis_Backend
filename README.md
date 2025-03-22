# AI Backend

This repository holds the AI backend for our smart code analysis project. The backend is built using FastAPI and serves as the API for the frontend. It interacts with the Qdrant vector database and the Ollama language model to provide intelligent responses to user queries.

We use business conext to provide the LLM with more context, this gives it an upperhand compared to cloud LLM's that don't have any business context

The backend can run as is and give functional resulsts, but its highly recommended to use our front end

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

http://localhost:8000/docs

This will open the Swagger-UI, where you can:

- View all available API endpoints.
- Test the endpoints directly from the browser.

--- 
# Technical details

## Diagram

![The Lab Architecture.png](readme-content/The%20Lab%20Architecture.png)

## Key components

- FastAPI Backend: Provides the API for interaction.
- Ollama Model: A custom-trained language model that understands the Machi Koro rulebook and generates intelligent responses.
- Qdrant Vector Database: Stores the embeddings of the business context  and enables efficient similarity searches.

### Contributors

[Jarne Aerts](https://www.linkedin.com/in/jarne-aerts/) \
[Rob Hellemans](https://www.linkedin.com/in/rob-hellemans/) \
[Seppe Van Hoof](https://www.linkedin.com/in/seppe-van-hoof-b76786225/)