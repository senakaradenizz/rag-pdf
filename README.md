# 📄 AI PDF Question Answering

A small local RAG-style application that lets users upload a PDF and ask grounded questions about its content.

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-App-red)
![Ollama](https://img.shields.io/badge/LLM-Ollama-black)
![License](https://img.shields.io/badge/License-MIT-green)

Upload a PDF, ask questions and receive AI-generated answers with page references.

![Application Demo](assets/app-demo.png)

## ✨ Features

🔍 Retrieves relevant PDF passages  
💬 Answers questions in natural language  
📖 Provides page references  
🔒 Runs locally with Ollama  

<details>
<summary>🚀 Click to view installation instructions</summary>
  
```bash
curl -fsSL https://ollama.com/install.sh | sh
```

```bash
pip install -r requirements.txt
streamlit run main.py
