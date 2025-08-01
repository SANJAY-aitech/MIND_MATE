
from flask import Flask, request, jsonify, render_template, send_file
from flask_cors import CORS
from pymongo import MongoClient
from werkzeug.security import generate_password_hash, check_password_hash
import os
from dotenv import load_dotenv
from datetime import datetime
import uuid
import io

# LangChain + Ollama
from langchain_ollama import ChatOllama
from langchain.schema.messages import AIMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder, PromptTemplate
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain.memory.chat_message_histories.in_memory import ChatMessageHistory

# HuggingFace Transformers for emotion detection
from transformers import pipeline

# Qdrant setup
from qdrant_client import QdrantClient
from qdrant_client.http.models import PointStruct, Filter, FieldCondition, MatchValue
from sentence_transformers import SentenceTransformer

load_dotenv()
app = Flask(__name__)
CORS(app)

# MongoDB only for credentials
client = MongoClient(os.getenv("MONGO_URI"))
db = client["mindmate"]
students = db["students"]

# Qdrant setup
qdrant = QdrantClient(
    url="https://your-qdrant-endpoint",  # Replace with actual
    api_key=os.getenv("QDRANT_API_KEY")
)
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
COLLECTION_NAME = "chat_history"
VECTOR_DIM = 384

if COLLECTION_NAME not in [c.name for c in qdrant.get_collections().collections]:
    qdrant.recreate_collection(
        collection_name=COLLECTION_NAME,
        vectors_config={"size": VECTOR_DIM, "distance": "Cosine"}
    )

llm = ChatOllama(model="gemma3:1b", temperature=0, max_tokens=50)

# Emotion classification
emotion_classifier = pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base", top_k=1)

prompt_template = ChatPromptTemplate.from_messages([
    ("system", """
You are MindMate, a calm, empathetic, and supportive emotional wellness AI designed to assist students experiencing academic, personal, or emotional stress.

--- Retrieved suggestions or past conversations ---
{context}

--- Emotional and conversational memory ---
{memory}
    """),
    MessagesPlaceholder(variable_name="messages"),
    ("human", "{user_input}")
])

user_histories = {}
def get_user_history(session_id: str):
    if session_id not in user_histories:
        user_histories[session_id] = ChatMessageHistory()
    return user_histories[session_id]

def detect_emotion(text):
    try:
        return emotion_classifier(text)[0][0]['label']
    except:
        return "unknown"

@app.route('/signup', methods=['POST'])
def signup():
    data = request.json
    if students.find_one({'email': data['email']}):
        return jsonify({'status': 'error', 'message': 'Email already registered'})
    hashed_pw = generate_password_hash(data['password'])
    students.insert_one({
        'name': data['name'],
        'email': data['email'],
        'password': hashed_pw,
        'created_at': datetime.utcnow()
    })
    return jsonify({'status': 'success'})

@app.route('/student-login', methods=['POST'])
def student_login():
    data = request.json
    student = students.find_one({'email': data['email']})
    if student and check_password_hash(student['password'], data['password']):
        email = student['email']
        name = student['name']
        result = qdrant.scroll(
            collection_name=COLLECTION_NAME,
            scroll_filter=Filter(
                must=[FieldCondition(key="email", match=MatchValue(value=email))]
            ),
            limit=5
        )
        points = result[0]
        if points:
            recent_emotion = points[0].payload.get("emotion", "neutral")
            past_tips = [
                p.payload["ai_response"]
                for p in points if any(kw in p.payload["ai_response"].lower() for kw in ["tip", "strategy", "suggest", "try"])
            ]
            tip_line = f" I previously suggested: “{past_tips[0]}” – did it help?" if past_tips else ""
            greeting = f"Hi again! Last time you felt {recent_emotion}. How are you feeling now?{tip_line}"
        else:
            greeting = "Welcome! I'm here to support you. 😊"

        return jsonify({
            'user': {
                'username': name,
                'email': email,
                'type': 'student',
                'greeting': greeting
            }
        })
    return jsonify({'error': 'Invalid email or password'})

@app.route('/chat', methods=['POST'])
def chat():
    data = request.get_json()
    user_input = data.get("message", "")
    user_email = data.get("email", "")

    if not user_email:
        return jsonify({"error": "Missing email"}), 400

    try:
        query_vector = embedding_model.encode(user_input).tolist()
        results = qdrant.search(
            collection_name=COLLECTION_NAME,
            query_vector=query_vector,
            limit=3,
            filter=Filter(
                must=[FieldCondition(key="email", match=MatchValue(value=user_email))]
            )
        )
        docs_text = "\n".join([
            f"You said: {r.payload['user_message']}\nI said: {r.payload['ai_response']}"
            for r in results if 'user_message' in r.payload and 'ai_response' in r.payload
        ])
        history = get_user_history(user_email)
        memory_text = "\n".join([
            f"User: {msg.content}" if isinstance(msg, HumanMessage) else f"AI: {msg.content}"
            for msg in history.messages
        ])
        inputs = {
            "context": docs_text,
            "memory": memory_text,
            "messages": history.messages,
            "user_input": user_input
        }

        chain = (prompt_template | llm | StrOutputParser())
        runnable = RunnableWithMessageHistory(
            chain,
            get_user_history,
            input_messages_key="user_input",
            history_messages_key="messages"
        )
        response = runnable.invoke(inputs, config={"configurable": {"session_id": user_email}})
        emotion = detect_emotion(user_input)

        qdrant.upload_points(
            collection_name=COLLECTION_NAME,
            points=[
                PointStruct(
                    id=str(uuid.uuid4()),
                    vector=query_vector,
                    payload={
                        "email": user_email,
                        "user_message": user_input,
                        "ai_response": response,
                        "emotion": emotion,
                        "type": "chat",
                        "timestamp": datetime.utcnow().isoformat()
                    }
                )
            ]
        )
        return jsonify({"response": response, "emotion": emotion})
    except Exception as e:
        print(f"Chat error: {e}")
        return jsonify({"response": "I'm sorry, something went wrong."})

@app.route('/history', methods=['GET'])
def get_history():
    user_email = request.args.get("email", "")
    if not user_email:
        return jsonify({"error": "Missing email"}), 400
    try:
        results = qdrant.scroll(
            collection_name=COLLECTION_NAME,
            scroll_filter=Filter(
                must=[FieldCondition(key="email", match=MatchValue(value=user_email))]
            ),
            limit=50
        )
        history = [
            {"user": p.payload.get("user_message", ""), "bot": p.payload.get("ai_response", "")}
            for p in results[0]
        ]
        return jsonify({"history": history})
    except Exception as e:
        print(f"History error: {e}")
        return jsonify({"history": []})

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "version": "2.0.0"
    })

if __name__ == '__main__':
    app.run(debug=True, port=5000)
