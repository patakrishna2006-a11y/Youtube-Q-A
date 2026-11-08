
from youtube_transcript_api import YouTubeTranscriptApi
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_nvidia_ai_endpoints import ChatNVIDIA
from langchain_classic.retrievers import MultiQueryRetriever
from langchain_core.output_parsers import StrOutputParser
from langchain_community.document_loaders.youtube import YoutubeLoader
from langchain_chroma import Chroma
from langchain_core.prompts import PromptTemplate
from langchain_ollama import OllamaEmbeddings

import dotenv
import os
import shutil

dotenv.load_dotenv()

# Step 1: Initialize LLM
llm = ChatNVIDIA(
    model="nvidia/nemotron-3-ultra-550b-a55b",
    temperature=0,
    max_completion_tokens=5000
)

# Step 2: Extract YouTube Video ID
url = input("Enter YouTube Link: ")

video_id = YoutubeLoader.extract_video_id(url)

# Step 3: Fetch YouTube Transcript
language = input("Enter language code (en, hi, fr, es): ")

youtube_api = YouTubeTranscriptApi()

transcript = youtube_api.fetch(
    video_id,
    languages=[language]
)

transcript_text = " ".join(
    snippet.text for snippet in transcript
)

# Step 4: Chunking
splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    separators=["\n\n", "\n", " ", ""]
)

chunks = splitter.create_documents(
    [transcript_text],
    metadatas=[{"video_id": video_id, "language": language}]
)

# Step 5: Embeddings
embedding = OllamaEmbeddings(
    model="embeddinggemma-2:latest"
)

# Step 6: Create Vector Store
if os.path.exists("Transcript.db"):
    shutil.rmtree("Transcript.db")

vector_store = Chroma(
    embedding_function=embedding,
    persist_directory="Transcript.db",
    collection_name="sample"
)

vector_store.add_documents(chunks)

# Step 7: MultiQuery Retriever
retriever = MultiQueryRetriever.from_llm(
    retriever=vector_store.as_retriever(
        search_kwargs={"k": 5}
    ),
    llm=llm
)

# Step 8: Prompt Template
prompt = PromptTemplate(
    template="""
You are an expert YouTube video assistant.

Answer the user's question using only the provided context.

Instructions:
- Explain all relevant concepts.
- For summaries, cover all major topics.
- Use clear headings and structured explanations.
- Include examples when supported by the context.
- Do not invent information.
- If the context is insufficient, clearly mention it.

Context:
{context}

Question:
{question}

Answer:
""",
    input_variables=["context", "question"]
)

# Step 9: LCEL Chain
chain = (
    prompt
    | llm
    | StrOutputParser()
)

# Step 10: Question Answering
while True:

    question = input("\nEnter your question: ")

    # Retrieve relevant documents
    retrieved_docs = retriever.invoke(question)

    # Convert documents into context text
    context = "\n\n".join(
        doc.page_content for doc in retrieved_docs
    )

    # Generate final answer
    result = chain.invoke({
        "context": context,
        "question": question
    })

    print("\n========== FINAL ANSWER ==========")
    print(result)