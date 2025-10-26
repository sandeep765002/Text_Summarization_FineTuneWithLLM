# Domain-Specific Text Summarizer with LLM Fine-Tuning

This project demonstrates the end-to-end development of a text summarization service powered by a fine-tuned Large Language Model (LLM). It showcases proficiency in deep learning, natural language processing (NLP), model fine-tuning, API development, and containerized deployment.

## Project Overview

The goal of this project is to create a tool capable of generating concise, high-quality summaries from long-form text. To achieve this, I fine-tuned an open-source T5 (Text-to-Text Transfer Transformer) model on a domain-specific dataset (e.g., news articles, scientific papers, legal documents – *customize this for your actual data*), and then exposed the model via a scalable FastAPI.

**Key Features:**
*   **Abstractive Summarization:** Generates novel summaries, not just extracting sentences.
*   **Domain Adaptation:** Fine-tuned on specific data for improved relevance and accuracy in a chosen domain.
*   **Scalable API:** Built with FastAPI, enabling easy integration into other applications.
*   **Containerized Deployment:** Uses Docker for reproducible and portable deployment.
*   **Performance Metrics:** Evaluated using standard ROUGE scores to quantify summarization quality.

## Technologies Used

*   **Python:** Core programming language.
*   **PyTorch:** Deep Learning framework.
*   **Hugging Face Transformers:** For LLM access (T5-small), tokenization, and fine-tuning utilities.
*   **Hugging Face Datasets:** For efficient data loading and preprocessing.
*   **FastAPI:** High-performance web framework for the API.
*   **Uvicorn:** ASGI server for running FastAPI.
*   **Docker:** For containerization.
*   **`rouge_score`:** For evaluating summarization quality.
*   **`python-dotenv`:** For environment variable management.

## Project Structure

text-summarizer-project/
├── data/ # Stores raw and processed datasets
├── models/ # Stores the fine-tuned T5 model and tokenizer
│ └── fine_tuned_t5/
├── src/ # Contains Python scripts for data, training, and prediction
│ ├── data_preparation.py # Script for dataset loading and tokenization
│ ├── model_training.py # Script for fine-tuning the T5 model
│ └── predict.py # Utility for local model inference
├── app/ # FastAPI application directory
│ ├── main.py # FastAPI application entry point
│ ├── requirements.txt # Python dependencies for the API
│ └── Dockerfile # Dockerfile for building the API image
├── .env # Environment variables for the API
├── README.md # Project README (this file)
└── requirements.txt # Python dependencies for the development/training environment
code
