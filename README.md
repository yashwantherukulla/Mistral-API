# Mistral API Docker Setup

This project sets up a Docker container to run a FastAPI application that uses the Mistral AI model for text generation. The application is designed to handle conversational queries and maintain a chat history.

## Table of Contents

- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Usage](#usage)
- [API Endpoints](#api-endpoints)
- [Dependencies](#dependencies)
- [Contributing](#contributing)
- [License](#license)

## Prerequisites

Before you begin, ensure you have the following installed:

- Docker
- Docker Compose (optional, for multi-container setups)
- A Hugging Face account and API token

## Installation

1. **Clone the repository:**

   ```bash
   git clone https://github.com/your-username/mistral-api.git
   cd mistral-api
   ```

2. **Build the Docker image:**

   Replace `YOUR_HUGGINGFACE_TOKEN` with your actual Hugging Face token.

   ```bash
   sudo docker build -t mistral-api --build-arg TOKEN=YOUR_HUGGINGFACE_TOKEN
   ```

3. **Run the Docker container:**

   ```bash
   sudo docker run -p 8000:8000 mistral-api
   ```

   The application will be accessible at `http://127.0.0.1:8000`.

## Usage

The application provides an endpoint to interact with the Mistral AI model. You can send a POST request to `/endpoint/` with a JSON payload containing your query.

### Example Request

```json
{
  "query": "What is the capital of France?"
}
```

### Example Response

```json
{
  "response": "The capital of France is Paris."
}
```

## API Endpoints

- **POST `/endpoint/`**: Send a query to the Mistral AI model and receive a response.

## Dependencies

The project uses several Python libraries, which are listed in the `requirements.txt` file. Key dependencies include:

- FastAPI: For building the API.
- Hugging Face Transformers: For loading and using the Mistral AI model.
- LangChain: For managing the conversation history and prompt templates.
