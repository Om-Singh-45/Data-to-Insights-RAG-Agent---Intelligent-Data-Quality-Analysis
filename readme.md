# Data-to-Insights — RAG Agent (Streamlit)

## Overview
A Streamlit app that lets you upload a CSV/Excel, indexes the dataset into a Chroma vectorstore with OpenAI embeddings, and answers natural-language questions using a Retrieval-Augmented Generation (RAG) workflow. The app also attempts to generate a simple chart for "top N by X" style questions.

Assignment spec (local copy): `/mnt/data/MYAIGURU_AI Engineers_Build Round Specs (Nov25_v01).pdf`

## Setup
1. Clone this repo.
2. Create virtual environment and install dependencies:
   ```bash
   python -m venv venv
   source venv/bin/activate   # or venv\\Scripts\\activate on Windows
   pip install -r requirements.txt
