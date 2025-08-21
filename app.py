import os
import json
import re
import time
import asyncio
from contextlib import asynccontextmanager
from dotenv import load_dotenv
from operator import itemgetter
import requests
from bs4 import BeautifulSoup

from fastapi import FastAPI, Depends, HTTPException, Header
from fastapi.responses import JSONResponse


from utils.DocsLoader import load_and_chunk
from utils.Schemas import RunRequest, RunResponse
# from concurrent.futures import ThreadPoolExecutor
from langchain_community.vectorstores import FAISS
from langchain.schema import Document
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings 
# from langchain_chroma import Chroma
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import CrossEncoderReranker 
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain.prompts import ChatPromptTemplate


from sentence_transformers import SentenceTransformer

#loading the model for SentenceTransformersTokenTextSplitter
MODEL_DIR = os.path.join("/tmp", "e5-large-v2")

if not os.path.exists(MODEL_DIR):
    print("📦 Downloading SentenceTransformer model...")
    model = SentenceTransformer("intfloat/e5-large-v2")
    model.save(MODEL_DIR)
    print("✅ Model saved at", MODEL_DIR)

    
# Load environment variables
load_dotenv()

vector_cache = {}
ml_models = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("🚀 Initializing models and prompt template...")

    try:
        GOOGLE_API_KEY = os.getenv("gemini_api_key3")
        print("🔑 gemini_api_key:", "FOUND" if GOOGLE_API_KEY else "NOT FOUND")

        if not GOOGLE_API_KEY:
            raise RuntimeError("CRITICAL: Missing GOOGLE_API_KEY in environment secrets!")

        
        # Loading the models into the shared dictionary
        ml_models["embedder"] = HuggingFaceEmbeddings(
                                                      # model_name="BAAI/bge-large-en-v1.5", #better but lil more slower
                                                      model_name="BAAI/bge-base-en-v1.5", #better but lil slower
                                                      # model_name="intfloat/e5-large-v2",
                                                      # encode_kwargs={
                                                      #     "batch_size": 64,
                                                      #     # "normalize_embeddings": True
                                                      # }
        # ml_models["embedder"] = NVIDIAEmbeddings(model="nvidia/nv-embedqa-e5-v5", nvidia_api_key=nvidia_api_key)
                                                    
    )
        cross_encoder_model = HuggingFaceCrossEncoder(model_name="BAAI/bge-reranker-base")
        
        ml_models["reranker_compressor"] = CrossEncoderReranker(model=cross_encoder_model, top_n=9)
        
        ml_models["llm"] = ChatGoogleGenerativeAI(
                               model="gemini-2.5-flash",                     
                               api_key=GOOGLE_API_KEY,
                           )
        
        # making the prompt (chain of thoughts)
        ml_models["prompt_template"] = ChatPromptTemplate.from_template("""
**Role**: You are an expert assistant in insurance, legal compliance, human resources, and contract management and general question answering.
**Instructions**:
Step 1 – **Initial Draft**:
- If the query contains multiple questions, split them into  perfect sub-questions.
- Use ONLY the provided context to answer.
- Provide one concise, complete sentence per sub-question.
- List answers in the same order as the sub-questions, without repeating the query text.
- Do not add numbering or bullet points; separate answers with a single space.
- Make the answer well structured and with proper starting like a human is answering it.
- Make grammatically correct sentence, improving phrasing and spelling.
- Avoid phrases like “the provided document states” or “according to the context.”
- Do NOT use line breakers ("/n" ,"\" and "/") in between the answers.
- Summarize relevant parts of the context without losing meaning.
- Avoid boilerplate phrases like “the document states” or “according to the context.”
- If the answer is not in the context for some subqueries, respond exactly with: " I do not know the answer of "subquery",Please ask query related to the Document only." for that subquery.
- Make sure that the You answer  the query in the same language in which the query is asked.
Step 2 – **Critique & Revise**:
- Review the initial answers for any missing or underused context.
- Revise responses to improve accuracy, completeness, grammar and clarity based on the context.
- Maintain a professional and domain-appropriate tone.
Step 3 – **Final Output**:
- Present the revised and cohesive set of responses.
---
**Context**:
{context}
---
**Query**:
{full_query}
---
**Response**:
"""
)

        
        print("✅ Models and prompt loaded successfully!")
    except Exception as e:
        print("❌ Lifespan error:", str(e))
        raise e
    
    yield
    print("🧹 Cleaning up.")
    ml_models.clear()
# --- 2. FastAPI App Instance ---
app = FastAPI(title="HackRX RAG Server", lifespan=lifespan)



# --- 3. API Key Verification ---
TEAM_API_KEY = os.getenv("TEAM_API_KEY")

def verify_api_key(authorization: str = Header(...)):
    if not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Invalid Authorization header format")
    token = authorization.split("Bearer ")[1]

    # 1st for initial team token , 2nd for real time token
    if token != TEAM_API_KEY:
        raise HTTPException(status_code=403, detail="Invalid or missing API key")



# --- 4. Main API Endpoint ---
@app.post("/api/v1/hackrx/run", response_model=RunResponse, dependencies=[Depends(verify_api_key)])
async def run_hackrx(req: RunRequest):
    
    doc_url = str(req.documents)
    lower_url = doc_url.lower()

    start_time = time.time()
    chunks = load_and_chunk(str(req.documents))
        
    if not chunks:
      return JSONResponse({"error": "No documents could be processed."}, status_code=400)    
            
    end_time = time.time() - start_time
        
    print(f"chunking done: {end_time}")
             
    start_time2 = time.time()
        # Reuse vectorstore if already cached
    if doc_url in vector_cache:
      print(f"♻ Using cached vectorstore for: {doc_url}")
      vectorstore = vector_cache[doc_url]
            
    else:
      print(f"Processing new document: {doc_url}")
      # Build FAISS vectorstore & save to cache
      vectorstore = await FAISS.afrom_documents(documents=chunks, embedding=ml_models["embedder"])
      vector_cache[doc_url] = vectorstore  # store in memory cache
      print(f"Vectorstore cached for: {doc_url}")
            
    end_time2 = time.time() - start_time2
    print(f"vector done: {end_time2}")

    dense_retriever = vectorstore.as_retriever(search_type="mmr",search_kwargs={"k": 14 ,"lambda_mult": 0.7} ) 
    # dense_retriever = vectorstore.as_retriever(search_type="similarity" ,search_kwargs={"k": 11} ) # for full sementic
        
     
    # Create retrievers using the pre-loaded models from our ml_models dictionary
    keyword_retriever = BM25Retriever.from_documents(chunks)
    keyword_retriever.k = 9 #prev 11
        
        # dense_retriever = Chroma.from_documents(documents=chunks, embedding=ml_models["embedder"]).as_retriever()
        
    ensemble_retriever = EnsembleRetriever(retrievers=[keyword_retriever, dense_retriever], weights=[0.3, 0.7],search_kwargs={"k": 14}) 
        
    compression_retriever = ContextualCompressionRetriever(
            base_retriever=ensemble_retriever, base_compressor=ml_models["reranker_compressor"]
        )
     
        # RAG chain 
    hybrid_rag_chain = (
            {"context": itemgetter("full_query") | ensemble_retriever, "full_query": itemgetter("full_query")}
            | ml_models["prompt_template"]
            | ml_models["llm"]
        )
     
    tasks = [hybrid_rag_chain.ainvoke({"full_query": q}) for q in req.questions]
    results = await asyncio.gather(*tasks)
       
    answers = []
    for msg in results:
            if hasattr(msg, "content"):
                answers.append(msg.content.strip())    
    
    return JSONResponse({"answers": answers}, status_code=200)

@app.get("/", include_in_schema=False)
def root():
    return {"message": "API is running."}
