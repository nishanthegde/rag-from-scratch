import os
from typing import List

import openai
import pinecone

OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]
PINECONE_API_KEY = os.environ["PINECONE_API_KEY"]
PINECONE_ENVIRONMENT = os.environ["PINECONE_ENVIRONMENT"]

# Global variable to maintain mapping
ID_TO_CHUNK_MAPPING = {}


def get_text(file_name: str) -> str:
  """Read the text file into memory."""
  with open(file_name, 'r') as f:
    text = f.read()

  return text


def chunk_text_by_characters(text: str, chunk_size: int) -> List[str]:
  return [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]


def embed_text(chunk: str) -> List[float]:
  response = openai.Embedding.create(input=chunk,
                                     model="text-embedding-ada-002")
  return response['data'][0]['embedding']


def store_embeddings_in_pinecone(embeddings: List[List[float]],
                                 chunks: List[str]) -> None:
  """Store embeddings in Pinecone and return a mapping of IDs to chunks."""

  vectors_to_upsert = []

  # Initialize Pinecone
  pinecone.init(api_key=PINECONE_API_KEY, environment=PINECONE_ENVIRONMENT)

  # Create or reference an index
  index_name = 'text-embeddings-index'

  if index_name not in pinecone.list_indexes():
    pinecone.create_index(index_name, dimension=1536)

  index = pinecone.Index(index_name)

  for idx, embedding in enumerate(embeddings):
    unique_id = f"chunk_{idx}"
    record_to_upsert = {}
    record_to_upsert['id'] = unique_id
    record_to_upsert['values'] = embedding
    vectors_to_upsert.append(record_to_upsert)

  # Populate the global mapping
  ID_TO_CHUNK_MAPPING.update(
      {f"chunk_{idx}": chunk
       for idx, chunk in enumerate(chunks)})

  index.upsert(vectors=vectors_to_upsert)


def get_text_from_id(chunk_id: str) -> str:
  return ID_TO_CHUNK_MAPPING.get(chunk_id, "")


def retrieve_similar_chunks(question: str) -> List[str]:
  embedding = embed_text(question)

  # Use the Pinecone index to find the most similar chunks of text
  index_name = 'text-embeddings-index'
  index = pinecone.Index(index_name)

  # print(embedding)
  # Adjusted for the newer Pinecone API
  results = index.query(vector=embedding, top_k=5)

  # Extract the IDs of the most similar chunks
  most_similar_chunk_ids = results['matches']
  # print(most_similar_chunk_ids)
  similar_chunks = [
      get_text_from_id(chunk['id']) for chunk in most_similar_chunk_ids
  ]

  return similar_chunks


def construct_prompt(question: str, similar_chunks: List[str]) -> str:
  # Joining the similar chunks with a newline for better readability
  formatted_chunks = "\n".join(similar_chunks)

  prompt = f"Context information is below.\n" \
          "---------------------\n" \
          f"{formatted_chunks}\n" \
          "---------------------\n" \
          "Given the context information and no prior knowledge, " \
          "answer the query.\n" \
          f"Query: {question}\n" \
          "Answer: "

  return prompt


def get_gpt_response(prompt: str) -> str:
  # Using the chat models with the v1/chat/completions endpoint
  response = openai.ChatCompletion.create(model="gpt-4",
                                          messages=[{
                                              "role":
                                              "system",
                                              "content":
                                              "You are a helpful assistant."
                                          }, {
                                              "role": "user",
                                              "content": prompt
                                          }])
  # Extracting the assistant's reply from the response
  assistant_message = response['choices'][0]['message']['content']

  return assistant_message


def main():
  text = get_text('input.txt')
  chunks = chunk_text_by_characters(text, 500)
  embeddings = [embed_text(chunk) for chunk in chunks]
  store_embeddings_in_pinecone(embeddings, chunks)

  while True:
    question = input("Ask a question: ")
    if question == "quit":
      break
    similar_chunks = retrieve_similar_chunks(question)
    prompt = construct_prompt(question, similar_chunks)
    response = get_gpt_response(prompt)
    print(f"Response: {response}")


if __name__ == '__main__':
  main()
