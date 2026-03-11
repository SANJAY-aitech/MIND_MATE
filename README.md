# 🧠 MindMate AI - Mental Wellness Companion

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10+-blue.svg" alt="Python">
  <img src="https://img.shields.io/badge/Flask-2.3+-green.svg" alt="Flask">
  <img src="https://img.shields.io/badge/LangChain-0.3+-orange.svg" alt="LangChain">
  <img src="https://img.shields.io/badge/MongoDB-6.0+-brightgreen.svg" alt="MongoDB">
  <img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="License">
</p>

> An intelligent mental wellness platform designed to support students experiencing academic, personal, or emotional stress. MindMate provides a private, judgment-free space for emotional expression with AI-powered conversations and mood tracking.

---

## 📖 What is MindMate AI?

MindMate AI is a comprehensive mental wellness platform that bridges the gap between students needing emotional support and educators monitoring student wellbeing. The application leverages modern AI technologies to provide personalized, context-aware emotional assistance while giving teachers valuable insights into student mental health trends.

### Core Philosophy

- **Privacy-First**: All conversations are private and stored securely
- **Non-Judgmental**: AI is designed to listen, not judge
- **Evidence-Based**: Uses RAG (Retrieval-Augmented Generation) for accurate suggestions
- **Accessible**: Supports voice input/output for hands-free interaction

---

## ✨ Features

### 👨‍🎓 Student Features

| Feature | Description |
|---------|-------------|
| **Secure Authentication** | Email/password registration with bcrypt password hashing |
| **Mood Tracking** | Select from 6 moods: Happy, Neutral, Sad, Anxious, Excited, Tired |
| **AI Chatbot** | Conversational AI powered by Ollama (gemma3:1b) with context awareness |
| **Voice Mode** | Speech-to-text input and text-to-speech output |
| **Chat History** | Persistent conversation history stored in MongoDB |
| **Journal Download** | Export conversations as downloadable text file |
| **Personalized Greetings** | AI remembers previous emotions and past suggestions |

### 👨‍🏫 Teacher Features

| Feature | Description |
|---------|-------------|
| **Dashboard Overview** | Real-time statistics on student wellness |
| **Mood Analytics** | Weekly mood distribution charts |
| **Check-in Trends** | Daily check-in frequency visualization |
| **Support Alerts** | Identify students needing immediate support |
| **Positive Metrics** | Track positive vs. concerning mood ratios |

---

## 🏗 Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           CLIENT LAYER                                   │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  │
│  │   Web UI    │  │ Voice Input │  │  Mood Input │  │ Charts/Stats │  │
│  │  (HTML/CSS) │  │(Speech API) │  │ (Selection) │  │  (Chart.js)  │  │
│  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘  │
└─────────┼────────────────┼────────────────┼────────────────┼─────────┘
          │                │                │                │
          ▼                ▼                ▼                ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                          API LAYER (Flask)                              │
│  ┌──────────────┐ ┌──────────────┐ ┌──────────────┐ ┌──────────────┐│
│  │  /signup     │ │  /chat        │ │ /submit-mood │ │/teacher-stats││
│  │  /login      │ │  /history     │ │ /download    │ │              ││
│  └──────────────┘ └──────────────┘ └──────────────┘ └──────────────┘│
└────────────────────────────────┬────────────────────────────────────────┘
                                 │
        ┌────────────────────────┼────────────────────────┐
        │                        │                        │
        ▼                        ▼                        ▼
┌───────────────┐     ┌─────────────────────┐    ┌───────────────┐
│   MongoDB     │     │  Pinecone Vector DB │    │  Ollama LLM   │
│  (User Data)  │     │  (RAG Retrieval)    │    │ (AI Response) │
└───────────────┘     └─────────────────────┘    └───────────────┘
                              │
                              ▼
                    ┌─────────────────────┐
                    │   HuggingFace       │
                    │   Embeddings        │
                    │ (Sentence-Transform)│
                    └─────────────────────┘
```

---

## 🛠 Tech Stack

### Backend

| Technology | Purpose | Version |
|------------|---------|---------|
| **Python** | Core language | 3.10+ |
| **Flask** | Web framework | 2.3+ |
| **Flask-CORS** | Cross-origin requests | - |
| **MongoDB** | Primary database | 6.0+ |
| **PyMongo** | MongoDB driver | - |

### AI & Machine Learning

| Technology | Purpose | Version |
|------------|---------|---------|
| **LangChain** | LLM orchestration & memory | 0.3+ |
| **LangChain-Ollama** | Ollama integration | - |
| **LangChain-Pinecone** | Vector store integration | - |
| **Ollama (gemma3:1b)** | Local LLM for AI responses | Latest |
| **Pinecone** | Vector database for RAG | - |
| **HuggingFace Embeddings** | Sentence transformers | - |
| **Transformers** | Emotion classification | - |

### Frontend

| Technology | Purpose |
|------------|---------|
| **HTML5** | Semantic markup |
| **CSS3** | Styling with animations |
| **JavaScript (ES6+)** | Client-side logic |
| **Chart.js** | Data visualization |
| **Marked.js** | Markdown parsing |
| **DOMPurify** | XSS protection |

---

## 📁 Project Structure

```
MIND_MATE/
│
├── app.py                      # Main Flask application (Pinecone version)
├── app_qdrant_updated.py      # Alternative version with Qdrant vector DB
├── ingestion.py               # Document ingestion pipeline for Pinecone
├── retrieval.py               # Retrieval testing script
├── requirements.txt           # Python dependencies
│
├── templates/
│   └── index.html             # Single-page application HTML
│
├── static/
│   ├── app.js                # Frontend JavaScript logic
│   └── styles.css            # CSS styling with animations
│
└── document/                  # (Optional) PDF documents for RAG
```

---

## 🔧 Installation

### Prerequisites

1. **Python 3.10+** installed
2. **MongoDB** instance (local or Atlas)
3. **Ollama** installed and running with gemma3:1b model
4. **Pinecone** account with API key

### Step 1: Clone and Setup

```bash
# Navigate to project directory
cd MIND_MATE-main

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Step 2: Install Ollama

```bash
# Install Ollama (macOS/Linux)
curl -fsSL https://ollama.com/install.sh | sh

# Pull the gemma3:1b model
ollama pull gemma3:1b

# Start Ollama server (default port 11434)
ollama serve
```

### Step 3: Create .env File

Create a `.env` file in the project root:

```env
# MongoDB Configuration
MONGO_URI=mongodb://localhost:27017/mindmate

# Pinecone Configuration
PINECONE_API_KEY=your_pinecone_api_key
PINECONE_INDEX_NAME=your_index_name

# Optional: For Qdrant version
QDRANT_API_KEY=your_qdrant_api_key
```

---

## ⚙️ Configuration

### Pinecone Setup

1. Create a Pinecone account at [pinecone.io](https://www.pinecone.io)
2. Create a new index with:
   - **Dimension**: 384 (for MiniLM embeddings)
   - **Metric**: Cosine
   - **Spec**: Serverless (AWS us-east-1 recommended)

### MongoDB Setup

```bash
# Option 1: Local MongoDB
# Install from https://www.mongodb.com/try/download/community

# Option 2: MongoDB Atlas (Cloud)
# Create free tier cluster and get connection string
```

### Document Ingestion (Optional RAG)

To enable context-aware responses from custom documents:

```bash
# Place PDF documents in the 'document/' folder
mkdir -p document
# Add your PDF files here

# Run ingestion pipeline
python ingestion.py
```

---

## 🚀 Running the Application

### Start Ollama (Terminal 1)

```bash
ollama serve
```

### Start Flask Application (Terminal 2)

```bash
# Ensure virtual environment is activated
python app.py
```

The application will start on `http://localhost:5000`

### Access the Application

Open your browser and navigate to:
```
http://localhost:5000
```

---

## 🔌 API Endpoints

### Authentication

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/signup` | Register new student |
| POST | `/student-login` | Student login |
| POST | `/teacher-login` | Teacher login |

### Chat & Mood

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/chat` | Send message to AI |
| POST | `/submit-mood` | Submit mood selection |
| GET | `/history` | Get chat history |
| GET | `/download-journal` | Download journal as file |

### Teacher Dashboard

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/teacher-stats` | Get aggregated statistics |

### Health Check

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/health` | API health status |

---

## 💾 Data Models

### MongoDB Collections

#### students
```json
{
  "_id": ObjectId,
  "name": "string",
  "email": "string",
  "password": "hashed_string",
  "created_at": datetime
}
```

#### journals
```json
{
  "_id": ObjectId,
  "email": "string",
  "entries": [{ "date": datetime, "content": "string" }]
}
```

#### chat_logs
```json
{
  "_id": ObjectId,
  "email": "string",
  "message": "string",
  "response": "string",
  "emotion": "string",
  "timestamp": datetime,
  "contains_tips": boolean
}
```

#### mood_logs
```json
{
  "_id": ObjectId,
  "email": "string",
  "name": "string",
  "mood": "string",
  "timestamp": datetime
}
```

---

## 🤖 AI & ML Pipeline

### Emotion Detection

```python
# Uses j-hartmann/emotion-english-distilroberta-base
emotion_classifier = pipeline(
    "text-classification", 
    model="j-hartmann/emotion-english-distilroberta-base",
    top_k=1
)
```

**Supported Emotions:**
- joy, sadness, anger, fear, surprise, disgust, neutral

### Retrieval-Augmented Generation (RAG)

```
User Input → Embed (MiniLM) → Pinecone Search → Context → LLM Prompt → Response
```

### Memory System

The application uses LangChain's `ChatMessageHistory` for:
- Session-based conversation context
- Past emotion tracking
- Tip/suggestion follow-ups

### Prompt Engineering

```python
system_prompt = """
You are MindMate, a calm, empathetic, and supportive emotional wellness AI...
- Act as a private, judgment-free listener
- Help students express feelings openly
- Guide in understanding stress, anxiety, emotional challenges
- Offer actionable suggestions from retrieved context
- Respect privacy and never escalate unless consent given
"""
```

---

## 🎨 Frontend Features

### Single-Page Application Flow

```
Landing → Role Selection → Login/Signup → Dashboard
```

### UI Components

| Component | Description |
|-----------|-------------|
| Mood Selector | 6-button mood selection with emoji icons |
| Chat Interface | Message bubbles with markdown support |
| Voice Mode | Animated microphone button with pulsing effect |
| History Panel | Collapsible previous conversation viewer |
| Stats Cards | Real-time metrics with icons |

### Animations

- Floating background shapes
- Smooth page transitions
- Mood button hover/select effects
- Voice mode pulse animation

---

## 📊 Teacher Dashboard

### Statistics Displayed

1. **Total Students** - Unique students with mood submissions (30 days)
2. **Most Common Mood** - Aggregated dominant emotion
3. **Positive Percentage** - Happy + Excited ratio
4. **Need Support** - Count of Sad + Anxious students

### Visualizations

| Chart | Type | Data Range |
|-------|------|------------|
| Weekly Mood Trends | Stacked Bar | Last 7 days by mood |
| Daily Check-ins | Line Chart | Last 14 days |

---

## 🔐 Security Considerations

- **Password Hashing**: Using bcrypt via `werkzeug.security`
- **Input Sanitization**: DOMPurify for XSS prevention
- **Environment Variables**: API keys stored in `.env`
- **CORS Configuration**: Limited to necessary origins

### Production Recommendations

1. Enable HTTPS/SSL
2. Implement JWT authentication
3. Add rate limiting
4. Use environment-specific configs
5. Implement audit logging

---

## 🧩 Troubleshooting

### Common Issues

| Issue | Solution |
|-------|----------|
| Ollama connection error | Ensure `ollama serve` is running |
| Pinecone connection failed | Verify API key and index name |
| MongoDB connection timeout | Check MongoDB URI format |
| Emotion detection fails | Check transformers package version |
| Voice mode not working | Use Chrome/Edge browser |

### Debug Mode

The application runs in debug mode by default. Check terminal for detailed error logs.

---

## 🚦 Future Enhancements

- [ ] Video consultation feature
- [ ] Crisis detection and intervention
- [ ] Multi-language support
- [ ] Mobile app (React Native)
- [ ] Anonymous mode
- [ ] Appointment scheduling with counselors
- [ ] Gamification for engagement

---

## 📄 License

MIT License - See [LICENSE](LICENSE) for details.

---

## 👏 Acknowledgments

- [LangChain](https://langchain.io) - AI orchestration
- [Ollama](https://ollama.com) - Local LLM
- [Pinecone](https://pinecone.io) - Vector database
- [HuggingFace](https://huggingface.co) - Pre-trained models

---

<p align="center">
  Made with ❤️ for student mental wellness
</p>

