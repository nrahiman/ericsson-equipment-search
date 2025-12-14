import streamlit as st
import requests
import pickle
import os
import numpy as np
import faiss

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
            
            embeddings = np.array([item["embedding"] for item in data]).astype('float32')
            index = faiss.IndexFlatL2(1536)
            index.add(embeddings)
            
            st.session_state.index = index
            st.session_state.data = data
            st.session_state.initialized = True
            st.success(f"✅ Loaded {len(data)} images!")
            st.rerun()
else:
    st.success("✅ Ready")
    query = st.text_input("🔍 Search:", placeholder="radio equipment")
    num_results = st.slider("Results:", 1, 5, 3)
    
    if st.button("Search") and query:
        with st.spinner("Searching..."):
            qemb = np.array([get_embedding(query)]).astype('float32')
            distances, indices = st.session_state.index.search(qemb, num_results)
            
            st.markdown("---")
            st.markdown(f"### Results: *{query}*")
            
            for idx in indices[0]:
                item = st.session_state.data[idx]
                col1, col2 = st.columns([1, 3])
                with col1:
                    try:
                        st.image(item['filename'], use_container_width=True)
                    except:
                        st.info("📷 Image")
                with col2:
                    st.markdown(f"**{item['filename']}**")
                    st.write(item['description'])
                st.markdown("---")
