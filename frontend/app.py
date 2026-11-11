import os

import requests
import streamlit as st

API_URL = os.getenv("API_URL", "http://localhost:8000").rstrip("/")

st.set_page_config(page_title="Food Classifier", page_icon="🍛")
st.title("🍛 Image Classifier")
st.caption("Upload a photo and the model will predict the top 3 classes.")

with st.sidebar:
    st.subheader("API status")
    st.code(API_URL)
    if st.button("Check API health"):
        try:
            r = requests.get(f"{API_URL}/health", timeout=60)
            r.raise_for_status()
            st.success(r.json())
        except requests.RequestException as e:
            st.error(f"API not reachable: {e}")

file = st.file_uploader("Drag and drop an image, or browse", type=["jpg", "jpeg", "png", "webp"])

if file:
    st.image(file, caption=file.name, use_container_width=True)
    if st.button("Classify", type="primary"):
        with st.spinner("Classifying... (the first request can take ~1 minute if the server was asleep)"):
            try:
                resp = requests.post(
                    f"{API_URL}/predict",
                    files={"file": (file.name, file.getvalue(), file.type)},
                    timeout=90,
                )
            except requests.Timeout:
                st.error("The server took too long to respond. It may be waking up, so please try again.")
            except requests.ConnectionError:
                st.error("Could not connect to the API. Check the API_URL setting.")
            else:
                if resp.ok:
                    result = resp.json()
                    st.subheader("Predictions")
                    for p in result["predictions"]:
                        st.write(f"**{p['label'].replace('_', ' ').title()}** : {p['confidence']*100:.1f}%")
                        st.progress(min(max(p["confidence"], 0.0), 1.0))
                    st.caption(f"Model version: {result['model_version']}")
                else:
                    try:
                        msg = resp.json().get("detail", resp.text)
                    except ValueError:
                        msg = resp.text
                    st.error(f"Error {resp.status_code}: {msg}")
