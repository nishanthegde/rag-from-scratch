# RAG-FROM-SCRATCH

This is a basic implementation of a bot for Retrieval Augmented Generation (RAG). The implementation reads and processes a file into tokenized chunks, embeds these chunks using OpenAI's Embedding API, and stores the embeddings in a vector database. It then takes a user question, finds the relevant content in the vector database, and sends it along to GPT-3.5 or GPT-4 to generate an answer.

## Authentication

To use the bot you need to authenticate as an end user. You need to create API keys for OPENAI as well as for the Pinecone vector database. 

Follow these links to get API authentication:
 - OPENAI: https://openai.com/product
 - Pinecone: https://docs.pinecone.io/docs/authentication