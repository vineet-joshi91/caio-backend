# -*- coding: utf-8 -*-
"""
Created on Wed Jul 30 12:14:44 2025

@author: Vineet
"""

# app.py

import streamlit as st
from backend.parser import parse_file
from backend.brains import analyze_cfo, analyze_cmo, analyze_coo, analyze_chro
from backend.config import ADMIN_EMAIL, FULL_USERS, DEMO_MODE, MAX_DOC_DOWNLOADS

st.set_page_config(page_title="CAIO – Chief AI Officer", layout="centered")

st.title("CAIO – Chief AI Officer for Business")
st.markdown("#### Instantly get tactical & strategic recommendations for your business – CFO, COO, CMO, CHRO in one click.")

# User session state
if 'downloads' not in st.session_state:
    st.session_state['downloads'] = 0
if 'user_email' not in st.session_state:
    st.session_state['user_email'] = ''
if 'llm_engine' not in st.session_state:
    st.session_state['llm_engine'] = 'dummy'
if 'api_key' not in st.session_state:
    st.session_state['api_key'] = ''

# Sidebar login/auth logic
def authenticate():
    st.sidebar.header("Login / Access")
    email = st.sidebar.text_input("Enter your email")
    key = st.sidebar.text_input("Enter your OpenAI API key (for full version)", type="password")
    if st.sidebar.button("Log In / Continue"):
        email_clean = email.strip().lower()
        st.session_state['user_email'] = email_clean
        st.session_state['api_key'] = key.strip()
        # Admin and full user check
        if email_clean == ADMIN_EMAIL or email_clean in FULL_USERS:
            if key:
                st.session_state['llm_engine'] = 'openai'
                import os
                os.environ["OPENAI_API_KEY"] = key.strip()
                st.sidebar.success("Full version unlocked. Enjoy real AI-powered insights!")
            else:
                st.session_state['llm_engine'] = 'dummy'
                st.sidebar.info("Admin access: Please enter your API key for live insights.")
        else:
            st.session_state['llm_engine'] = 'dummy'
            st.sidebar.warning("Demo version only. Enter a registered email/API key for full access.")

authenticate()

uploaded_file = st.file_uploader("Upload your business document", type=['pdf', 'xlsx', 'xls', 'docx', 'txt'])
st.markdown("**OR**")
query = st.text_area("Paste your business question or data here (no document required):")

if st.button("Analyze"):
    # Check demo download limit
    if st.session_state['llm_engine'] == 'dummy' and DEMO_MODE and st.session_state['downloads'] >= MAX_DOC_DOWNLOADS:
        st.error(f"Demo usage limit reached! Contact us for full access.")
        st.stop()

    if uploaded_file:
        text = parse_file(uploaded_file)
        input_label = f"Uploaded file: {uploaded_file.name}"
    elif query.strip():
        text = query.strip()
        input_label = "Your text query"
    else:
        st.warning("Please upload a document or enter a query.")
        st.stop()

    with st.spinner("Analyzing with CAIO..."):
        st.markdown(f"### Input Analyzed: {input_label}")
        st.subheader("CFO Insights")
        st.write(analyze_cfo(text))
        st.subheader("COO Insights")
        st.write(analyze_coo(text))
        st.subheader("CMO Insights")
        st.write(analyze_cmo(text))
        st.subheader("CHRO Insights")
        st.write(analyze_chro(text))

        if st.session_state['llm_engine'] == 'dummy':
            st.info("This is a demo output. Some features are restricted. Contact for full version.")

        st.session_state['downloads'] += 1

st.markdown("---")
st.caption("© 2025 CAIO. For demo use only. Contact for commercial access.")
