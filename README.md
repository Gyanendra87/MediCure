# 🩺 MediCure – AI-Powered Medical Chatbot

MediCure is an intelligent medical assistant that combines **Generative AI (Google Gemini)**, **Machine Learning (XGBoost)**, and **Natural Language Processing** to answer health-related questions and suggest **home remedies** for common conditions.

---

## 🚀 Features

- 💬 **Medical Question Answering** using Google's Gemini API and medical knowledge base
- 🌿 **Home Remedies Prediction** using a trained ML model (TF-IDF + XGBoost)
- 📚 **Knowledge Base Retrieval** via FAISS vector database
- 🧠 **Hybrid Intelligence**: Combines rule-based retrieval + AI reasoning
- ⚡ **FastAPI Backend** ready for deployment
- 🎯 **LangGraph Workflow** for intelligent query routing

---

## 🧩 Tech Stack

| Category | Tools / Libraries |
|----------|------------------|
| **Language** | Python 3.8+ |
| **AI Model** | Google Gemini 2.5 Flash (via `langchain-google-genai`) |
| **Embeddings** | `sentence-transformers/all-MiniLM-L6-v2` |
| **Vector Store** | FAISS |
| **ML Model** | XGBoost + TF-IDF Vectorizer |
| **Framework** | LangChain, LangGraph |
| **Data Sources** | `medical.txt`, `home_remedies.csv` |
| **Backend** | FastAPI (assumed from your setup) |
| **Environment** | `.env` for API keys |

---

## 🧠 How It Works

1. **Load Environment Variables**  
   Loads your `GOOGLE_API_KEY` from `.env` file

2. **Document Loading & Splitting**  
   Loads `medical.txt` and splits into chunks for efficient retrieval

3. **Create FAISS Vector Database**  
   Stores medical text embeddings for semantic search

4. **Train Home Remedy Model**  
   Uses `home_remedies.csv` to train a TF-IDF + XGBoost classifier for remedy prediction

5. **LangGraph Chat Flow**  
   Implements a two-node graph:
   - **retrieve**: Fetches relevant medical context
   - **generate**: Uses Gemini or the remedy predictor based on query type

---

## 🧰 Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager
- Google API key for Gemini

### Setup Steps

1. **Clone the repository**
```bash
git clone https://github.com/Gyanendra87/MediCure.git
cd MediCure
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Create `.env` file**  
Create a `.env` file in the root directory:
```env
GOOGLE_API_KEY=your_google_api_key_here
```

---

## ▶️ Running the Project

### Start the backend server:
```bash
uvicorn app:app --reload
```

The server will start at `http://127.0.0.1:8000`

### Access the frontend:
Open `frontend/index.html` in your browser or navigate to the appropriate endpoint.

---

## 🧩 Project Structure

```
MediCure/
│
├── app.py                      # FastAPI backend server
├── graph_builder.py            # LangGraph workflow definition
├── check.py                    # Utility functions
│
├── data/
│   └── medical.txt             # Medical knowledge base
│
├── home_remedies.csv           # Home remedies dataset
├── medical_book.pdf            # Additional medical reference
│
├── frontend/
│   └── index.html              # Simple web interface
│
├── requirements.txt            # Python dependencies
├── .env                        # Environment variables (not committed)
└── README.md                   # This file
```

---

## 🧪 Example Queries

### Home Remedy Request
**User:** "What is a home remedy for indigestion?"  
**Bot:** 🌿 Home Remedy for Indigestion: Use ginger — helps soothe the stomach and improve digestion.

### Medical Question
**User:** "Explain what causes ear infections."  
**Bot:** 💬 Ear infections are often caused by bacteria or viruses in the middle ear, commonly following a cold or respiratory infection...

### Symptom-based Query
**User:** "I have a headache and fever"  
**Bot:** Provides relevant medical information and suggests appropriate remedies.

---

## 📦 Dependencies

Key libraries used in this project:

- `fastapi` - Web framework
- `langchain` - LLM orchestration
- `langchain-google-genai` - Google Gemini integration
- `langchain-huggingface` - Embeddings
- `langgraph` - Graph-based workflow
- `faiss-cpu` - Vector similarity search
- `xgboost` - Machine learning
- `scikit-learn` - ML preprocessing
- `pandas` - Data manipulation
- `python-dotenv` - Environment management

---

## 🔑 API Keys

You'll need a Google API key to use Gemini:

1. Visit [Google AI Studio](https://makersuite.google.com/app/apikey)
2. Create or select a project
3. Generate an API key
4. Add it to your `.env` file

---

## 📈 Future Enhancements

- 🧬 Add symptom-based disease prediction
- 🔊 Voice-based query support
- 📱 Mobile app integration
- 🩹 Integration with wearable sensors
- 🌍 Multi-language support
- 📊 User health tracking dashboard
- 🔒 Enhanced privacy and data security

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 👨‍💻 Author

**Gyanendra Singh**  
📘 B.Tech in Electronics & Communication Engineering, IIIT Una  
🔗 [GitHub Profile](https://github.com/Gyanendra87)

---

## ⚠️ Disclaimer

**Important:** This chatbot is designed for **educational and informational purposes only**.

- It does **NOT** replace professional medical advice, diagnosis, or treatment
- Always consult a qualified healthcare provider for medical concerns
- In case of emergency, contact your local emergency services immediately

---

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

---

## 🙏 Acknowledgments

- Google Gemini for powerful AI capabilities
- LangChain community for excellent tools
- Open-source contributors of all used libraries

---

**Made with ❤️ by Gyanendra Singh**
