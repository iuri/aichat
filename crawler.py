import os
from playwright.sync_api import sync_playwright
from bs4 import BeautifulSoup
from urllib.parse import urlparse, urljoin

from sentence_transformers import SentenceTransformer
import chromadb
import uuid
import pickle

import numpy as np
import faiss

# Step 1. Retrieve data from source 
visited = set()
crawled_data = {}

def is_internal_link(link, base_domain):
    parsed_link = urlparse(link)
    return parsed_link.netloc == "" or base_domain in parsed_link.netloc

def extract_links_from_html(html, base_url, base_domain):
    soup = BeautifulSoup(html, 'html.parser')
    links = set()
    for a_tag in soup.find_all('a', href=True):
        href = a_tag['href']
        full_url = urljoin(base_url, href)
        if is_internal_link(full_url, base_domain):
            links.add(full_url.split("#")[0])
    return links

def crawl_website(start_url, max_pages=20):
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()
        base_domain = urlparse(start_url).netloc
        to_visit = [start_url]

        while to_visit and len(visited) < max_pages:
            current_url = to_visit.pop(0)
            if current_url in visited:
                continue
            print(f"🔍 Crawling: {current_url}")
            visited.add(current_url)

            try:
                page.goto(current_url, timeout=60000)
                page.wait_for_timeout(3000)  # Wait for content to render
                html = page.content()
                soup = BeautifulSoup(html, 'html.parser')
                for script in soup(["script", "style", "noscript"]): script.decompose()
                text = soup.get_text(separator="\n", strip=True)
                crawled_data[current_url] = text

                new_links = extract_links_from_html(html, current_url, base_domain)
                for link in new_links:
                    if link not in visited and link not in to_visit:
                        to_visit.append(link)

            except Exception as e:
                print(f"⚠️ Error crawling {current_url}: {e}")
        
        browser.close()
    return crawled_data


def get_text_from_url(url):
    # Start crawling
    knowledge_base = crawl_website(url, max_pages=10)

    # Save knowledge base to a file
    with open("amx_knowledge_base.txt", "w", encoding="utf-8") as f:
        for url, text in knowledge_base.items():
            f.write(f"\n\n=== {url} ===\n{text}\n")

    return "amx_knowledge_base.txt"








# Step 2 
def chunk_text(text, chunk_size=500):
    sentences = text.split(". ")
    chunks, current = [], ""
    for sentence in sentences:
        if len(current) + len(sentence) < chunk_size:
            current += sentence + ". "
        else:
            chunks.append(current.strip())
            current = sentence + ". "
    chunks.append(current.strip())
    return chunks


# Step 3
def get_embedding(text, model="text-embedding-ada-002"):
    response = openai.Embedding.create(
        input=text,
        model=model
    )
    return response['data'][0]['embedding']






if __name__ == "__main__":
    # Step 1: Extract Content from AMX Healthcare Website
    filename = "amx_knowledge_base.txt"
    if not os.path.exists(filename):
        filename = get_text_from_url("https://www.amxhealthcare.com/")

    print("Contentsuccessfully retrieved!")
        
   
    # Step 2: Chunk the Text Into Semantic Chunks
    with open(filename, "r") as file:
        text = file.read()
    chunks = chunk_text(text, chunk_size=500)
    print(f"Extracted {len(chunks)} chunks from the text.")
    
    # Save for Flask
    with open("chunks.pkl", "wb") as f:
        pickle.dump(chunks, f)


    
    # Step 3: Convert Chunks to Embeddings
    model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = model.encode(chunks)


    with open("embeddings.npy", "wb") as f:
        np.save(f, embeddings)

    # Build and save FAISS index
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(np.array(embeddings).astype("float32"))
        



    
    # Step 4: Store Chunks + Embeddings in Vector DB
    chroma_client = chromadb.Client()
    collection = chroma_client.create_collection(name="amx_kb")

    for chunk, embedding in zip(chunks, embeddings):
        collection.add(documents=[chunk], embeddings=[embedding], ids=[str(uuid.uuid4())])




    



