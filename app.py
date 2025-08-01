from flask import Flask, request, jsonify, render_template, send_file
from flask_cors import CORS
from pymongo import MongoClient
from werkzeug.security import generate_password_hash, check_password_hash
import os
from dotenv import load_dotenv
from datetime import datetime, timedelta
import json
import io
from collections import Counter

# LangChain + Ollama + Pinecone
from langchain_ollama import ChatOllama
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone
from langchain.schema.messages import AIMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder,PromptTemplate

# New LangChain memory system
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain.memory.chat_message_histories.in_memory import ChatMessageHistory

# HuggingFace Transformers for emotion detection
from transformers import pipeline
import re

load_dotenv()
app = Flask(__name__)
CORS(app)

# === MongoDB Setup === #
client = MongoClient(os.getenv("MONGO_URI"))
db = client["mindmate"]
students = db["students"]
journals = db["journals"]
chat_logs = db["chat_logs"]
mood_logs = db["mood_logs"]  # NEW: Store mood submissions

# === Teacher Credentials === #
TEACHER_ID = "teacher123"
TEACHER_PASSWORD = "password123"

# === Pinecone + LangChain Setup === #
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index = pc.Index(os.getenv("PINECONE_INDEX_NAME"))

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vector_store = PineconeVectorStore(index=index, embedding=embeddings)
retriever = vector_store.as_retriever(search_type="similarity_score_threshold", search_kwargs={"k": 3, "score_threshold": 0.5})

llm = ChatOllama(model="gemma3:1b", temperature=0, max_tokens=50)

# Emotion classification pipeline
emotion_classifier = pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base", top_k=1)
greeting_prompt = PromptTemplate.from_template(
    """Hello! In our previous session, you were feeling {emotion}. 
I'm here to continue supporting you. How are you feeling today?

Respond in a warm, caring tone as a therapeutic AI assistant:
- Acknowledge their previous emotion
- Show genuine concern
- Keep it to 1-2 sentences
- Invite them to share how they're feeling now"""
)
def generate_ai_greeting(last_emotion):
    input_text = greeting_prompt.format(emotion=last_emotion)
    response = llm.invoke(input_text)
    return response.content.strip()

def detect_emotion(text):
    try:
        result = emotion_classifier(text)[0][0]
        emotion = result['label'].lower()

        # Override if message clearly implies stress
        stress_keywords = ["stress", "overwhelm", "too much", "anxious", "exam", "pressure", "tension"]
        if any(word in text.lower() for word in stress_keywords):
            if emotion == "joy":
                return "stress"
        return emotion
    except Exception as e:
        print(f"Emotion detection failed: {e}")
        return "unknown"


# === Prompt Template === #
prompt_template = ChatPromptTemplate.from_messages([
    ("system", """
You are MindMate, a calm, empathetic, and supportive emotional wellness AI designed to assist students experiencing academic, personal, or emotional stress.

Your mission:
- Act as a private, judgment-free listener and reflection partner.
- Help students express their feelings openly and safely.
- Guide students in understanding and managing stress, anxiety, and emotional challenges.
- Encourage self-awareness, resilience, and mindfulness practices.
- Offer actionable suggestions only if included in the retrieved context below.
- When user returns after previous session where you provided tips/suggestions, ask if they tried them and how it went.
- Respect privacy and never escalate concerns unless the user explicitly consents.
- Avoid clinical diagnoses, assumptions, or invented facts.
- Vary your tone and responses; be genuinely present in each interaction.

Always speak with:
- Warmth, understanding, and patience.
- Non-judgmental and non-intrusive language.
- A focus on listening over fixing.

--- Retrieved suggestions, reflections, or resources (if any) ---
{context}

--- Recent emotional and conversational memory ---
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

@app.route('/')
def index():
    return render_template('index.html')

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

        # Check if this is first login
        first_login = not chat_logs.find_one({'email': email})
        
        if first_login:
            greeting = "Welcome to MindMate! I'm here to support you. How are you feeling today?"
        else:
            # Fetch latest emotion from chat logs
            memory = chat_logs.find_one({'email': email}, sort=[('timestamp', -1)])
            last_emotion = memory.get('emotion') if memory else None

            # Check if we gave any tips in last conversation
            last_convo = list(chat_logs.find({'email': email}).sort('timestamp', -1).limit(10))
            tips_given = any(msg.get("contains_tips") is True for msg in last_convo)
            # Generate greeting based on last emotion and tips
            if last_emotion and last_emotion != "unknown":
                if tips_given:
                    # Look for the most recent tip
                    recent_tip_msg = next((msg for msg in last_convo if msg.get("contains_tips") is True), None)

                    if recent_tip_msg:
                        # Pick first line that looks like an actual tip (starts with "*", "-", or number)
                        tip_lines = [line.strip() for line in recent_tip_msg['response'].split('\n') if line.strip()]
                        tip_text = next((line for line in tip_lines if line.strip().startswith(("*", "-", "1", "•"))), None)
                        tip_text = tip_text[:100] if tip_text else "some techniques to reduce stress"


                        greeting = f"Welcome back! Last time you were feeling {last_emotion}. " \
                                    f"I suggested: \"{tip_text}\". Did you give it a try? I'm here if you want to talk more about it."

                    else:
                        greeting = generate_ai_greeting(last_emotion)
                else:
                    greeting = f"Welcome back! How are you feeling today? Last time you were {last_emotion}."
            else:
                greeting = "Welcome back! How are you feeling today?"

        return jsonify({
            'user': {
                'username': name,
                'email': email,
                'type': 'student',
                'greeting': greeting,
                'firstLogin': first_login
            }
        })

    return jsonify({'error': 'Invalid email or password'})


@app.route('/teacher-login', methods=['POST'])
def teacher_login():
    data = request.json
    if data['teacherId'] == TEACHER_ID and data['password'] == TEACHER_PASSWORD:
        return jsonify({'user': {'id': TEACHER_ID, 'type': 'teacher'}})
    return jsonify({'error': 'Invalid Teacher ID or password'})

# NEW: Submit Mood Endpoint
@app.route('/submit-mood', methods=['POST'])
def submit_mood():
    data = request.json
    mood_entry = {
        'email': data['email'],
        'name': data['name'],
        'mood': data['mood'],
        'timestamp': datetime.utcnow()
    }
    mood_logs.insert_one(mood_entry)
    return jsonify({'status': 'success', 'message': 'Mood submitted successfully'})

# NEW: Teacher Stats Endpoint
@app.route('/teacher-stats', methods=['GET'])
def get_teacher_stats():
    try:
        # Get all mood entries from the last 30 days
        thirty_days_ago = datetime.utcnow() - timedelta(days=30)
        recent_moods = list(mood_logs.find({'timestamp': {'$gte': thirty_days_ago}}))
        
        # Total unique students
        unique_students = len(set(mood['email'] for mood in recent_moods))
        
        # Count moods
        mood_counts = Counter(mood['mood'] for mood in recent_moods)
        total_moods = sum(mood_counts.values()) if mood_counts else 1
        
        # Most common mood
        most_common = mood_counts.most_common(1)
        most_common_mood = get_mood_emoji(most_common[0][0]) if most_common else "😊"
        
        # Positive responses (happy + excited)
        positive_count = mood_counts.get('happy', 0) + mood_counts.get('excited', 0)
        positive_percentage = round((positive_count / total_moods) * 100) if total_moods > 0 else 85
        
        # Students needing support (sad + anxious)
        support_moods = [mood for mood in recent_moods if mood['mood'] in ['sad', 'anxious']]
        students_need_support = []
        for mood in support_moods:
            students_need_support.append({
                'name': mood['name'],
                'email': mood['email'],
                'mood': mood['mood'],
                'moodEmoji': get_mood_emoji(mood['mood'])
            })
        
        # Remove duplicates by email
        unique_support = {student['email']: student for student in students_need_support}
        students_need_support = list(unique_support.values())
        
        # Weekly mood trends (last 7 days)
        weekly_data = get_weekly_mood_data(recent_moods)
        
        # Daily check-ins (last 14 days)
        daily_data = get_daily_checkin_data(recent_moods)
        
        return jsonify({
            'totalStudents': unique_students or 127,  # Default for demo
            'mostCommonMood': most_common_mood,
            'positivePercentage': positive_percentage,
            'studentsNeedSupport': students_need_support,
            'weeklyMoods': weekly_data,
            'dailyCheckins': daily_data
        })
    
    except Exception as e:
        print(f"Error in teacher stats: {e}")
        # Return demo data if database fails
        return jsonify({
            'totalStudents': 127,
            'mostCommonMood': '😊',
            'positivePercentage': 85,
            'studentsNeedSupport': [],
            'weeklyMoods': get_demo_weekly_data(),
            'dailyCheckins': get_demo_daily_data()
        })

def get_mood_emoji(mood):
    mood_emojis = {
        'happy': '😊',
        'neutral': '😐',
        'sad': '😢',
        'anxious': '😰',
        'excited': '🤩',
        'tired': '😴'
    }
    return mood_emojis.get(mood, '😊')

def get_weekly_mood_data(mood_entries):
    # Get data for last 7 days
    days = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
    weekly_counts = {mood: [0] * 7 for mood in ['happy', 'neutral', 'sad', 'anxious', 'excited', 'tired']}
    
    for mood_entry in mood_entries:
        day_index = mood_entry['timestamp'].weekday()  # 0=Monday, 6=Sunday
        mood_type = mood_entry['mood']
        if mood_type in weekly_counts:
            weekly_counts[mood_type][day_index] += 1
    
    return {
        'labels': days,
        'happy': weekly_counts['happy'],
        'neutral': weekly_counts['neutral'],
        'sad': weekly_counts['sad'],
        'anxious': weekly_counts['anxious'],
        'excited': weekly_counts['excited'],
        'tired': weekly_counts['tired']
    }

def get_daily_checkin_data(mood_entries):
    # Get check-in counts for last 14 days
    daily_counts = {}
    for mood_entry in mood_entries:
        date_str = mood_entry['timestamp'].strftime('%b %d')
        daily_counts[date_str] = daily_counts.get(date_str, 0) + 1
    
    # Create ordered list of last 14 days
    dates = []
    counts = []
    for i in range(13, -1, -1):
        date = datetime.utcnow() - timedelta(days=i)
        date_str = date.strftime('%b %d')
        dates.append(date_str)
        counts.append(daily_counts.get(date_str, 0))
    
    return {
        'dates': dates,
        'counts': counts
    }

def get_demo_weekly_data():
    return {
        'labels': ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'],
        'happy': [12, 15, 18, 14, 20, 8, 6],
        'neutral': [8, 10, 12, 11, 9, 15, 12],
        'sad': [3, 2, 4, 5, 3, 7, 8],
        'anxious': [5, 4, 6, 8, 7, 4, 5],
        'excited': [8, 12, 10, 6, 15, 3, 2],
        'tired': [4, 7, 5, 8, 6, 13, 17]
    }

def get_demo_daily_data():
    dates = []
    counts = []
    for i in range(13, -1, -1):
        date = datetime.utcnow() - timedelta(days=i)
        dates.append(date.strftime('%b %d'))
        counts.append(25 + (i % 10))  # Demo data
    
    return {
        'dates': dates,
        'counts': counts
    }

@app.route('/chat', methods=['POST'])
def chat():
    data = request.get_json()
    user_input = data.get("message", "")
    user_email = data.get("email", "")

    if not user_email:
        return jsonify({"error": "Missing email"}), 400

    try:
        docs = retriever.invoke(user_input)
        docs_text = "\n".join([doc.page_content[:300] for doc in docs]) if docs else ""

        history = get_user_history(user_email)
        memory_text = "\n".join([
            f"User: {msg.content}" if isinstance(msg, HumanMessage) else f"AI: {msg.content}"
            for msg in history.messages
        ])

        # Check if this is follow-up about a previous tip
        is_follow_up = any("tip:" in msg.content.lower() or 
                          "suggestion:" in msg.content.lower()
                          for msg in history.messages if isinstance(msg, AIMessage))

        recent_logs = list(chat_logs.find({"email": user_email}).sort("timestamp", -1).limit(3))
        past_emotions = [log.get("emotion") for log in recent_logs if log.get("emotion") != "unknown"]
        past_feelings = ", ".join(past_emotions[::-1]) if past_emotions else "neutral"

        # Add follow-up context to the main context if needed
        if is_follow_up:
            docs_text = "FOLLOW-UP CONTEXT: User is returning after previous session where suggestions were provided. " \
                      "Gently ask if they tried the suggestions and how it went.\n\n" + docs_text

        inputs = {
            "context": docs_text,
            "memory": memory_text,
            "messages": history.messages,
            "user_input": user_input,
            "past_feelings": past_feelings,
            "is_follow_up": is_follow_up
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
        
        timestamp = datetime.utcnow()

        # Mark if response contains tips/suggestions for future reference
       
        TIP_REGEX = re.compile(r"\b(tip|tips|suggestion|suggestions|try this|try doing|you can try|here are a few)\b", re.IGNORECASE)

        contains_tips = bool(TIP_REGEX.search(response)) if isinstance(response, str) else False


        chat_logs.insert_one({
            "email": user_email,
            "message": user_input,
            "response": response,
            "emotion": emotion,
            "timestamp": timestamp,
            "contains_tips": contains_tips
        })

        return jsonify({"response": response, "emotion": emotion})

    except Exception as e:
        print(f"Chat error: {e}")
        return jsonify({"response": "I'm sorry, I'm having trouble responding right now. Please try again."})
    
@app.route('/history', methods=['GET'])
def get_history():
    user_email = request.args.get("email", "")
    if not user_email:
        return jsonify({"error": "Missing email"}), 400

    try:
        # Get chat history from MongoDB instead of vector store for reliability
        chat_history = list(chat_logs.find({"email": user_email}).sort("timestamp", 1).limit(50))
        
        history = []
        for entry in chat_history:
            history.append({
                "user": entry.get("message", ""),
                "bot": entry.get("response", "")
            })

        return jsonify({"history": history})

    except Exception as e:
        print(f"History error: {e}")
        return jsonify({"history": []})

@app.route('/download-journal', methods=['GET'])
def download_journal():
    email = request.args.get("email", "")
    if not email:
        return jsonify({"error": "Missing email"}), 400

    try:
        entries = list(chat_logs.find({"email": email}).sort("timestamp", 1))
        if not entries:
            return jsonify({"error": "No journal entries found."}), 404

        content = f"MindMate AI - Personal Journal\nUser: {email}\nGenerated: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
        content += "="*50 + "\n\n"
        
        content += "\n\n".join([
            f"Date: {entry['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}\n"
            f"Emotion Detected: {entry.get('emotion', 'unknown').title()}\n"
            f"You: {entry['message']}\n"
            f"MindMate: {entry['response']}\n"
            f"{'-'*30}"
            for entry in entries
        ])

        buffer = io.BytesIO()
        buffer.write(content.encode('utf-8'))
        buffer.seek(0)

        return send_file(
            buffer,
            as_attachment=True,
            download_name=f"mindmate_journal_{email.split('@')[0]}_{datetime.utcnow().strftime('%Y%m%d')}_journal.txt.txt",
            mimetype='text/plain'
        )

    except Exception as e:
        print(f"Download error: {e}")
        return jsonify({"error": "Failed to generate journal"}), 500

# Health check endpoint
@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "version": "1.0.0"
    })

# Error handlers
@app.errorhandler(404)
def not_found(error):
    return jsonify({"error": "Endpoint not found"}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({"error": "Internal server error"}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)