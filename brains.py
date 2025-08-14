# -*- coding: utf-8 -*-
"""
Created on Wed Jul 30 12:04:44 2025

@author: Vineet
"""
# brains.py

from backend.engines import get_llm_response
import streamlit as st

def analyze_cfo(text):
    return get_llm_response(
        f"You are the CFO. Analyze the following business information and give sharp financial insights and tactical recommendations:\n\n{text}",
        engine=st.session_state.get('llm_engine', 'dummy')
    )

def analyze_cmo(text):
    return get_llm_response(
        f"You are the CMO. Analyze the following business information and give actionable marketing insights and growth strategies:\n\n{text}",
        engine=st.session_state.get('llm_engine', 'dummy')
    )

def analyze_coo(text):
    return get_llm_response(
        f"You are the COO. Analyze the following business information and give operational improvements, risk flags, and process optimizations:\n\n{text}",
        engine=st.session_state.get('llm_engine', 'dummy')
    )

def analyze_chro(text):
    return get_llm_response(
        f"You are the CHRO. Analyze the following business information and give people/HR strategy, cultural recommendations, and compliance insights:\n\n{text}",
        engine=st.session_state.get('llm_engine', 'dummy')
    )
