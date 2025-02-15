# app.py
from flask import Flask, render_template, request, jsonify, session
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
import logging
from datetime import datetime
import sqlite3
import uuid

# Konfigurasi logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.secret_key = 'your-secret-key-here'  # Ganti dengan secret key yang aman
app.config['PERMANENT_SESSION_LIFETIME'] = 86400  # Session berlaku 24 jam

# Konfigurasi Model
OLLAMA_CONFIG = {
    "model": "qwen",
    "base_url": "http://<ip-server-ollama>:11434",
    "temperature": 0.7,
    "top_k": 40,
    "top_p": 0.95,
    "num_ctx": 2048,
    "repeat_penalty": 1.1
}

# Inisialisasi database
def init_db():
    conn = sqlite3.connect('chat_history.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS sessions
                 (session_id TEXT PRIMARY KEY, created_at TIMESTAMP)''')
    c.execute('''CREATE TABLE IF NOT EXISTS messages
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  session_id TEXT,
                  role TEXT,
                  content TEXT,
                  timestamp TIMESTAMP,
                  FOREIGN KEY (session_id) REFERENCES sessions(session_id))''')
    conn.commit()
    conn.close()

# Initialize database at startup
init_db()

def get_system_prompt() -> SystemMessage:
    return SystemMessage(content="""Kamu adalah asisten guru matematika yang sabar dan membantu.
Panduan mengajar:
1. Jelaskan konsep dengan sederhana dan mudah dipahami
2. Berikan langkah penyelesaian step by step
3. Sertakan contoh yang relevan
4. Jika siswa bingung, coba pendekatan penjelasan yang berbeda
5. Dorong siswa untuk berpikir kritis
Berikan jawaban dalam bahasa Indonesia yang jelas, singkat dan edukatif dengan contoh soal jika diperlukan.""")

def save_message(session_id, role, content):
    """Menyimpan pesan ke database"""
    conn = sqlite3.connect('chat_history.db')
    c = conn.cursor()
    c.execute('INSERT INTO messages (session_id, role, content, timestamp) VALUES (?, ?, ?, ?)',
              (session_id, role, content, datetime.now()))
    conn.commit()
    conn.close()

def get_chat_history_from_db(session_id):
    """Mengambil chat history dari database"""
    conn = sqlite3.connect('chat_history.db')
    c = conn.cursor()
    c.execute('SELECT role, content, timestamp FROM messages WHERE session_id = ? ORDER BY timestamp', (session_id,))
    messages = [{"role": role, "content": content, "timestamp": str(timestamp)} 
                for role, content, timestamp in c.fetchall()]
    conn.close()
    return messages

def clear_chat_history(session_id):
    """Menghapus chat history dari database"""
    conn = sqlite3.connect('chat_history.db')
    c = conn.cursor()
    c.execute('DELETE FROM messages WHERE session_id = ?', (session_id,))
    conn.commit()
    conn.close()

# Inisialisasi LLM
try:
    llm = ChatOllama(**OLLAMA_CONFIG)
    logger.info("LLM initialized successfully")
except Exception as e:
    logger.error(f"Error initializing LLM: {str(e)}")
    llm = None

@app.route('/')
def home():
    # Buat session ID baru jika belum ada
    if 'session_id' not in session:
        session.permanent = True
        session_id = str(uuid.uuid4())
        session['session_id'] = session_id
        
        # Simpan session baru ke database
        conn = sqlite3.connect('chat_history.db')
        c = conn.cursor()
        c.execute('INSERT INTO sessions (session_id, created_at) VALUES (?, ?)',
                  (session_id, datetime.now()))
        conn.commit()
        conn.close()
    
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    if llm is None:
        return jsonify({'error': 'LLM not initialized properly'}), 500
    
    try:
        data = request.json
        user_input = data['message']
        session_id = session.get('session_id')
        
        if not session_id:
            return jsonify({'error': 'Invalid session'}), 400
        
        if user_input.lower() == 'clear':
            clear_chat_history(session_id)
            return jsonify({
                'response': 'Chat history cleared',
                'chat_log': []
            })
        
        # Simpan pesan user
        save_message(session_id, "user", user_input)
        
        # Ambil seluruh history untuk konteks
        chat_history = get_chat_history_from_db(session_id)
        
        # Konversi pesan untuk LLM
        llm_messages = [
            get_system_prompt(),
            *[HumanMessage(content=msg["content"]) if msg["role"] == "user" 
              else AIMessage(content=msg["content"]) 
              for msg in chat_history if msg["role"] in ["user", "assistant"]]
        ]
        
        # Generate response
        response = llm.invoke(llm_messages)
        
        # Simpan response asisten
        save_message(session_id, "assistant", response.content)
        
        # Ambil history terbaru
        updated_history = get_chat_history_from_db(session_id)
        
        return jsonify({
            'response': response.content,
            'chat_log': updated_history
        })
    
    except Exception as e:
        logger.error(f"Error in chat endpoint: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/get_chat_history', methods=['GET'])
def get_chat_history():
    session_id = session.get('session_id')
    if not session_id:
        return jsonify([])
    return jsonify(get_chat_history_from_db(session_id))

@app.route('/health')
def health_check():
    return jsonify({
        'status': 'healthy',
        'llm_status': 'initialized' if llm else 'not initialized',
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
