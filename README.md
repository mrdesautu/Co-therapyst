# 🤖 Co-Therapyst

**Co-Therapyst** is an AI-assisted platform designed to support psychologists in analyzing therapy sessions with clarity and insight.  
Built with modern LLM technology, this tool extracts clinical insights from session transcripts to enhance therapeutic decision-making.

---

## 🧠 Features

- Analyze therapy sessions from text or video (e.g., YouTube).
- Extract hypotheses, therapeutic goals, and emotional profiles.
- Track session progress using visual dashboards.
- Create a clinical knowledge base from multiple sessions.
- Generate automatic summaries and treatment suggestions.
- Use multiple AI chains and theoretical perspectives (in progress).

---

## ⚙️ Tech Stack

- **LangChain** for LLM orchestration and memory
- **ChromaDB** for semantic search and vector storage
- **OpenAI GPT / Claude** (via LangChain)
- **Streamlit** for front-end interface
- **Plotly & Matplotlib** for visualizations
- **HuggingFace** embeddings & Whisper ASR

---

## 🚀 How to Run

1. Clone this repository  
```bash
git clone https://github.com/your-username/cotherapyst.git
cd cotherapyst
```

2. Create and activate a virtual environment  
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies  
```bash
pip install -r requirements.txt
```

4. Set your environment variables (in `.env` file)  
```
OPENAI_API_KEY=your-key
HUGGINGFACEHUB_API_TOKEN=your-token
```

5. Run Streamlit app  
```bash
streamlit run app.py
```

---

## 📁 Project Structure

```
cotherapyst/
├── app.py                      # Streamlit front-end
├── llm_chains.py              # LLM prompt chains and agents
├── video-youtube-sesion.py    # Process sessions from YouTube
├── chroma_ingest.py           # Build vector DB from sessions
├── data/                      # Session text or audio
├── outputs/                   # Generated reports & logs
├── .env                       # API keys
└── README.md
```

---

## 🧪 Coming Soon

- Agent review pipelines per theoretical framework
- Multi-session longitudinal analysis
- Patient-friendly summaries and visual feedback
- Bedrock + AWS integration

---

## 🤝 Contributing

Pull requests are welcome! If you have ideas for improving psychological insights with LLMs, feel free to open an issue.

---

## 📜 License

MIT License. See `LICENSE` for more info.
