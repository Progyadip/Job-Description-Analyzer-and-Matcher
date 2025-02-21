import time
import warnings
import numpy as np
import pandas as pd
import faiss
import requests
import os
from sentence_transformers import SentenceTransformer
from huggingface_hub import login
from googlesearch import search
from dotenv import load_dotenv  # Securely load environment variables

warnings.filterwarnings("ignore")

# Load Environment Variables
load_dotenv()
HUGGINGFACE_API_KEY = os.getenv("HUGGINGFACE_API_KEY")

if not HUGGINGFACE_API_KEY:
    raise ValueError("Error: Hugging Face API key is missing. Add it to the .env file.")

login(token=HUGGINGFACE_API_KEY)  # Authenticate with Hugging Face

# Hugging Face Inference API URL
API_URL = "https://api-inference.huggingface.co/models/tiiuae/falcon-7b-instruct"
HEADERS = {"Authorization": f"Bearer {HUGGINGFACE_API_KEY}"}

# File path for resumes
file_path = r"D:\Desktop\job_finder_ai\resumes.csv"
print(f"File Exists: {os.path.exists(file_path)}")

# Load resumes from CSV
def load_resumes(file_path):
    try:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Error: File '{file_path}' not found.")

        if os.stat(file_path).st_size == 0:
            raise ValueError("Error: The CSV file is empty.")

        df = pd.read_csv(file_path, encoding="utf-8")

        if "resume_text" not in df.columns:
            raise ValueError("Error: Column 'resume_text' not found in the CSV file.")

        return df["resume_text"].dropna().str.strip().tolist()

    except Exception as e:
        print(f"Error loading resumes: {e}")
        return []

# Fetch job listings from Google Search (Indeed)
def get_indeed_jobs(query, num_jobs=5):
    job_links = []
    try:
        for url in search(f"{query} site:indeed.com", num_results=num_jobs, lang="en"):
            job_links.append(url)
    except Exception as e:
        print(f"Error fetching jobs: {e}")
    return job_links

# Text Preprocessing with Sentence Embeddings
embedding_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

def get_text_embeddings(text_list):
    return np.array(embedding_model.encode(text_list)).astype("float32")

# Store job descriptions in FAISS
def create_faiss_index(job_texts):
    if not job_texts:
        raise ValueError("Error: No job descriptions found. FAISS index cannot be created.")

    job_embeddings = get_text_embeddings(job_texts)
    index = faiss.IndexFlatL2(job_embeddings.shape[1])
    index.add(job_embeddings)
    return index, job_embeddings

# Find best matching jobs for a resume
def match_resume_to_jobs(resume_text, job_texts, faiss_index, job_embeddings):
    resume_embedding = get_text_embeddings([resume_text])
    _, indices = faiss_index.search(resume_embedding, k=3)
    return [job_texts[i] for i in indices[0]]

# Explain matches using Hugging Face Inference API
def explain_match(resume_text, job_description):
    prompt = f"Explain why this resume: '{resume_text}' is a good match for this job: '{job_description}'."
    response = requests.post(API_URL, headers=HEADERS, json={"inputs": prompt})

    try:
        json_response = response.json()
        if isinstance(json_response, list) and "generated_text" in json_response[0]:
            return json_response[0]["generated_text"]
        else:
            return f"Unexpected response format: {json_response}"
    except Exception as e:
        return f"Error processing response: {e}"

# Main Execution
if __name__ == "__main__":
    # Step 1: Fetch job listings from Google (Indeed)
    job_descriptions = get_indeed_jobs("Python Developer jobs", num_jobs=5)

    if not job_descriptions:
        print("Error: No job descriptions found. Check Indeed search.")
        exit()

    # Step 2: Load resumes
    resumes = load_resumes(file_path)  

    if not resumes:
        print("Error: No resumes found. Check resumes.csv file.")
        exit()

    # Step 3: Create FAISS index
    faiss_index, job_embeddings = create_faiss_index(job_descriptions)

    # Step 4: Match each resume and generate explanations
    for resume in resumes:
        matched_jobs = match_resume_to_jobs(resume, job_descriptions, faiss_index, job_embeddings)
        print("\n=== Matching Jobs for Resume ===")
        for job in matched_jobs:
            explanation = explain_match(resume, job)
            print(f"Job: {job}\nExplanation: {explanation}\n")

