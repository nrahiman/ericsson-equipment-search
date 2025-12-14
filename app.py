import streamlit as st
import requests
import chromadb
import pickle
import os

st.set_page_config(page_title="Ericsson Equipment Search", page_icon="📡", layout="wide")

API_KEY = "9sCteGVmfDgbLsSu5V4znBU3bmB08BfjmisPWAuJEHRx0njOZdSMJQQJ99BLACHYHv6XJ3w3AAAAACOGgxKS"
ENDPOINT = "https://najr-miyonro1-eastus2.cognitiveservices.azure.com/"

st.title("📡 Ericsson Equipment Search")
st.markdown("AI-powered semantic search")
st.markdown("---")

def get_embedding(text):
    url = f"{ENDPOINT}openai/deployments/text-embedding-ada-002/embeddings?api-version=2024-02-01"
    headers = {"api-key": API_KEY, "Content-Type": "application/json"}
    return requests.post(url, headers=headers, json={"input": text}).json()["data"][0]["embedding"]

if 'initialized' not in st.session_state:
    st.session_state.initialized = False

if not st.session_state.initialized:
    if st.button("🚀 Load Database"):
        with st.spinner("Loading..."):
            with open("ericsson_image_data.pkl", "rb") as f:
                data = pickle.load(f)
            chroma = chromadb.Client()
            try:
                chroma.delete_collection("demo")
            except:
                pass
            col = chroma.create_collection("demo")
            for i, item in enumerate(data):
                col.add(embeddings=[item["embedding"]], documents=[item["description"]], metadatas=[{"filename": item["filename"]}], ids=[f"img_{i}"])
            st.session_state.collection = col
            st.session_state.initialized = True
            st.success(f"✅ Loaded {len(data)} images!")
            st.rerun()
else:
    st.success("✅ Ready")
    query = st.text_input("🔍 Search:", placeholder="radio equipment")
    num_results = st.slider("Results:", 1, 5, 3)
    if st.button("Search") and query:
        with st.spinner("Searching..."):
            qemb = get_embedding(query)
            results = st.session_state.collection.query(query_embeddings=[qemb], n_results=num_results)
            st.markdown("---")
            st.markdown(f"### Results: *{query}*")
            for i, (doc, meta) in enumerate(zip(results['documents'][0], results['metadatas'][0])):
                col1, col2 = st.columns([1, 3])
                with col1:
                    if os.path.exists(meta['filename']):
                        st.image(meta['filename'], use_container_width=True)
                with col2:
                    st.markdown(f"**{meta['filename']}**")
                    st.write(doc)
                st.markdown("---")
