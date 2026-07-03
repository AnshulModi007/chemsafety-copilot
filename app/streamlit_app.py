"""Streamlit frontend for ChemSafety Copilot. Talks to the FastAPI backend over
HTTP (BACKEND_URL env var, default http://localhost:8000) rather than importing
the agent in-process -- this is a separate deployable service, matching the
brief's "FastAPI backend + Streamlit frontend" architecture.

Run locally with: streamlit run app/streamlit_app.py
(with the backend already running: uvicorn app.main:app)
"""
import os

import requests
import streamlit as st

BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")

st.set_page_config(page_title="ChemSafety Copilot", page_icon="⚗️")
st.title("ChemSafety Copilot")
st.caption(
    "Ask about past CSB chemical incidents, look up a chemical's properties, "
    "or size a relief valve. Not a substitute for a licensed Professional Engineer."
)

if "history" not in st.session_state:
    st.session_state.history = []

query = st.chat_input("Ask a question...")

for turn in st.session_state.history:
    with st.chat_message(turn["role"]):
        st.write(turn["content"])

if query:
    st.session_state.history.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.write(query)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                resp = requests.post(f"{BACKEND_URL}/ask", json={"query": query}, timeout=120)
                resp.raise_for_status()
                result = resp.json()
            except requests.RequestException as e:
                st.error(f"Couldn't reach the backend at {BACKEND_URL}: {e}")
                st.session_state.history.append({"role": "assistant", "content": f"Error: {e}"})
            else:
                st.write(result["answer"])
                st.caption(f"Routed as: **{result['intent']}** — {result['routing_reasoning']}")

                citations = result.get("data", {}).get("citations")
                if citations:
                    with st.expander(f"Sources ({len(citations)})"):
                        for c in citations:
                            st.write(f"- {c['report_id']}, page {c['page']}")

                st.session_state.history.append({"role": "assistant", "content": result["answer"]})
