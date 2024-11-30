from flask import Flask, request, jsonify, send_file
from werkzeug.utils import secure_filename
import os
from gtts import gTTS
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import mysql.connector
from mysql.connector import Error
import fitz  # PyMuPDF for PDF processing
from docx import Document

# Constants
ALLOWED_EXTENSIONS = {'txt', 'pdf', 'docx'}
UPLOAD_FOLDER = 'uploads'
VOICE_FOLDER = 'voice_messages'
AUDIO_FOLDER = 'audio_files'
DOCUMENT_FOLDER = 'documents'

# Database connection
def get_db_connection():
    return mysql.connector.connect(
        host="localhost",
        user="root",  # Your MySQL username
        password="root",  # Your MySQL password
        database="dylexsia"
    )

# Ensure necessary tables exist
def create_tables():
    try:
        conn = get_db_connection()
        cursor = conn.cursor()

        # Create table for storing document comparisons
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS document_comparisons (
                id INT AUTO_INCREMENT PRIMARY KEY,
                doc1_text TEXT,
                doc2_text TEXT,
                similarity_percentage FLOAT
            )
        """)

        # Create table for storing voice files
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS voice_files (
                id INT AUTO_INCREMENT PRIMARY KEY,
                filename VARCHAR(255),
                file_path VARCHAR(255)
            )
        """)

        # Create table for storing transcribed text
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS transcribed_text (
                id INT AUTO_INCREMENT PRIMARY KEY,
                text TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Create table for storing Speech-to-Text data
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS speech_to_text (
                id INT AUTO_INCREMENT PRIMARY KEY,
                audio_file_path VARCHAR(255),
                transcribed_text TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        conn.commit()
        cursor.close()
        conn.close()

    except Error as e:
        print(f"Error creating tables: {e}")

# Ensure folders exist
def create_folders():
    if not os.path.exists(UPLOAD_FOLDER):
        os.makedirs(UPLOAD_FOLDER)
    if not os.path.exists(VOICE_FOLDER):
        os.makedirs(VOICE_FOLDER)
    if not os.path.exists(AUDIO_FOLDER):
        os.makedirs(AUDIO_FOLDER)
    if not os.path.exists(DOCUMENT_FOLDER):
        os.makedirs(DOCUMENT_FOLDER)

# App class
class App:
    def __init__(self, name):
        self.app = Flask(name)
        # Create tables if they don't exist
        create_tables()
        # Create folders if they don't exist
        create_folders()
        
        # Register routes
        self.app.add_url_rule('/compare', 'compare_documents', self.compare_documents, methods=['POST'])
        self.app.add_url_rule('/convert', 'convert_text_to_speech', self.convert_text_to_speech, methods=['POST'])
        self.app.add_url_rule('/voice/<filename>', 'serve_voice', self.serve_voice, methods=['GET'])
        self.app.add_url_rule('/voices', 'get_all_voice_files', self.get_all_voice_files, methods=['GET'])
        self.app.add_url_rule('/transcribe', 'transcribe_audio', self.transcribe_audio, methods=['POST'])
        self.app.add_url_rule('/get_transcribed_text', 'get_transcribed_text', self.get_transcribed_text, methods=['GET'])
        self.app.add_url_rule('/delete_transcribed_text', 'delete_transcribed_text', self.delete_transcribed_text, methods=['DELETE'])
        self.app.add_url_rule('/update_transcribed_text', 'update_transcribed_text', self.update_transcribed_text, methods=['PUT'])

        # New Speech-to-Text endpoint
        self.app.add_url_rule('/speech_to_text', 'handle_speech_to_text', self.handle_speech_to_text, methods=['GET', 'POST', 'PUT', 'DELETE'])

    def run(self, debug=True):
        self.app.run(debug=debug, host='0.0.0.0', port=5000)

    # Document Comparison Endpoint
    def compare_documents(self):
        try:
            doc1_file = request.files.get('document1')
            doc2_file = request.files.get('document2')

            if not doc1_file or not doc2_file:
                return jsonify({'error': 'Both documents must be provided'}), 400

            doc1_path = self.save_file(doc1_file, DOCUMENT_FOLDER)
            doc2_path = self.save_file(doc2_file, DOCUMENT_FOLDER)

            doc1 = self.extract_text(doc1_path, doc1_file.filename)
            doc2 = self.extract_text(doc2_path, doc2_file.filename)

            similarity_percentage = self.compare_texts(doc1, doc2)

            # Save the comparison result to the database
            conn = get_db_connection()
            cursor = conn.cursor()
            cursor.execute("INSERT INTO document_comparisons (doc1_text, doc2_text, similarity_percentage) VALUES (%s, %s, %s)",
                           (doc1, doc2, similarity_percentage))
            conn.commit()
            cursor.close()
            conn.close()

            return jsonify({'similarity_percentage': similarity_percentage})

        except Exception as e:
            return jsonify({'error': str(e)}), 500

    # Helper for text preprocessing
    def preprocess_text(self, text):
        """Preprocess text for comparison"""
        return text.lower()

    # Helper to compare texts
    def compare_texts(self, doc1, doc2):
        """Compare two texts using cosine similarity"""
        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform([doc1, doc2])
        cosine_sim = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])
        similarity_percentage = cosine_sim[0][0] * 100
        return similarity_percentage

    # Text-to-Speech Endpoint
    def convert_text_to_speech(self):
        try:
            if 'file' not in request.files:
                return jsonify({'error': 'No file part'}), 400

            file = request.files['file']
            if file.filename == '':
                return jsonify({'error': 'No selected file'}), 400

            if not self.allowed_file(file.filename):
                return jsonify({'error': 'Unsupported file format'}), 400

            file_path = self.save_file(file, UPLOAD_FOLDER)
            input_text = self.extract_text(file_path, file.filename)

            output_file_path = self.convert_to_speech(input_text, VOICE_FOLDER, os.path.splitext(file.filename)[0])

            # Save voice file info to the database
            conn = get_db_connection()
            cursor = conn.cursor()
            cursor.execute("INSERT INTO voice_files (filename, file_path) VALUES (%s, %s)",
                           (os.path.basename(output_file_path), output_file_path))
            conn.commit()
            cursor.close()
            conn.close()

            if output_file_path:
                return send_file(output_file_path, as_attachment=True)
            else:
                return jsonify({'error': 'Failed to generate MP3'}), 500

        except Exception as e:
            return jsonify({'error': str(e)}), 500

    # Helper to check if file is allowed
    def allowed_file(self, filename):
        """Check if the file has an allowed extension"""
        return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

    # Helper to save files
    def save_file(self, file, folder):
        """Save the uploaded file and return the file path"""
        if not os.path.exists(folder):
            os.makedirs(folder)
        filename = secure_filename(file.filename)
        file_path = os.path.join(folder, filename)
        file.save(file_path)
        return file_path

    # Helper to extract text from files
    def extract_text(self, file_path, filename):
        """Extract text from the uploaded file"""
        input_text = ""
        if filename.endswith('.txt'):
            with open(file_path, 'r', encoding='utf-8') as f:
                input_text = f.read()
        elif filename.endswith('.pdf'):
            pdf_text = []
            with fitz.open(file_path) as pdf:
                for page in pdf:
                    pdf_text.append(page.get_text())
            input_text = '\n'.join(pdf_text)
        elif filename.endswith('.docx'):
            doc = Document(file_path)
            input_text = '\n'.join(paragraph.text for paragraph in doc.paragraphs)
        return input_text

    # Helper to convert text to speech
    def convert_to_speech(self, text, output_folder, filename):
        """Convert text to speech and save as an MP3 file"""
        try:
            tts = gTTS(text=text, lang='en')  # Change the language as needed
            output_file = os.path.join(output_folder, f'{filename}.mp3')
            tts.save(output_file)
            return output_file
        except Exception as e:
            print(f"Error generating MP3: {e}")
            return None

    # Serve Voice File Endpoint
    def serve_voice(self, filename):
        try:
            conn = get_db_connection()
            cursor = conn.cursor()
            cursor.execute("SELECT file_path FROM voice_files WHERE filename = %s", (filename,))
            file_path = cursor.fetchone()

            if file_path:
                return send_file(file_path[0])
            else:
                return jsonify({"error": "File not found"}), 404

        except Exception as e:
            return jsonify({"error": str(e)}), 500

    # Get All Voice Files Endpoint
    def get_all_voice_files(self):
        try:
            conn = get_db_connection()
            cursor = conn.cursor()
            cursor.execute("SELECT filename, file_path FROM voice_files")
            files = cursor.fetchall()

            return jsonify({'files': [{'filename': file[0], 'file_path': file[1]} for file in files]})

        except Exception as e:
            return jsonify({"error": str(e)}), 500

    # Transcribe Audio Endpoint
    def transcribe_audio(self):
        try:
            file = request.files.get('audio')
            if not file:
                return jsonify({"error": "No audio file provided"}), 400

            file_path = self.save_file(file, AUDIO_FOLDER)
            transcribed_text = self.transcribe_audio_file(file_path)

            conn = get_db_connection()
            cursor = conn.cursor()
            cursor.execute("INSERT INTO transcribed_text (text) VALUES (%s)", (transcribed_text,))
            conn.commit()
            cursor.close()
            conn.close()

            return jsonify({"message": "Audio transcribed successfully", "transcribed_text": transcribed_text})

        except Exception as e:
            return jsonify({"error": str(e)}), 500

    def transcribe_audio_file(self, file_path):
        # Your transcription logic here
        return "Transcribed text from audio"

    # Get Transcribed Text Endpoint
    def get_transcribed_text(self):
        try:
            conn = get_db_connection()
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM transcribed_text ORDER BY created_at DESC LIMIT 1")
            transcribed_text = cursor.fetchone()

            if transcribed_text:
                return jsonify({"id": transcribed_text[0], "text": transcribed_text[1]})
            else:
                return jsonify({"error": "No transcribed text found"}), 404

        except Exception as e:
            return jsonify({"error": str(e)}), 500

    # Delete Transcribed Text Endpoint
    def delete_transcribed_text(self):
        try:
            text_id = request.args.get('id')
            if not text_id:
                return jsonify({"error": "No ID provided"}), 400

            conn = get_db_connection()
            cursor = conn.cursor()
            cursor.execute("DELETE FROM transcribed_text WHERE id = %s", (text_id,))
            conn.commit()
            cursor.close()
            conn.close()

            return jsonify({"message": "Text deleted successfully"})

        except Exception as e:
            return jsonify({"error": str(e)}), 500

    # Update Transcribed Text Endpoint
    def update_transcribed_text(self):
        try:
            text_id = request.args.get('id')
            new_text = request.json.get('text')

            if not text_id or not new_text:
                return jsonify({"error": "Missing ID or text"}), 400

            conn = get_db_connection()
            cursor = conn.cursor()
            cursor.execute("UPDATE transcribed_text SET text = %s WHERE id = %s", (new_text, text_id))
            conn.commit()
            cursor.close()
            conn.close()

            return jsonify({"message": "Text updated successfully"})

        except Exception as e:
            return jsonify({"error": str(e)}), 500

    # Speech-to-Text Endpoint
    def handle_speech_to_text(self):
        if request.method == 'POST':
            return self.create_speech_to_text()
        elif request.method == 'GET':
            return self.get_speech_to_text()
        elif request.method == 'PUT':
            return self.update_speech_to_text()
        elif request.method == 'DELETE':
            return self.delete_speech_to_text()

    def create_speech_to_text(self):
        try:
            file = request.files.get('audio')
            if not file:
                return jsonify({"error": "No audio file provided"}), 400

            file_path = self.save_file(file, AUDIO_FOLDER)
            transcribed_text = self.transcribe_audio_file(file_path)

            conn = get_db_connection()
            cursor = conn.cursor()
            cursor.execute("INSERT INTO speech_to_text (audio_file_path, transcribed_text) VALUES (%s, %s)", (file_path, transcribed_text))
            conn.commit()
            cursor.close()
            conn.close()

            return jsonify({"message": "Audio transcribed successfully", "transcribed_text": transcribed_text})

        except Exception as e:
            return jsonify({"error": str(e)}), 500

    def get_speech_to_text(self):
        try:
            conn = get_db_connection()
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM speech_to_text ORDER BY created_at DESC LIMIT 1")
            record = cursor.fetchone()
            cursor.close()
            conn.close()

            if record:
                return jsonify({"id": record[0], "audio_file_path": record[1], "transcribed_text": record[2]})
            else:
                return jsonify({"error": "No records found"}), 404

        except Exception as e:
            return jsonify({"error": str(e)}), 500

    def update_speech_to_text(self):
        try:
            text_id = request.json.get('id')
            new_text = request.json.get('transcribed_text')

            if not text_id or not new_text:
                return jsonify({"error": "Missing ID or transcribed text"}), 400

            conn = get_db_connection()
            cursor = conn.cursor()
            cursor.execute("UPDATE speech_to_text SET transcribed_text = %s WHERE id = %s", (new_text, text_id))
            conn.commit()
            cursor.close()
            conn.close()

            return jsonify({"message": "Text updated successfully"})

        except Exception as e:
            return jsonify({"error": str(e)}), 500

    def delete_speech_to_text(self):
        try:
            text_id = request.args.get('id')
            if not text_id:
                return jsonify({"error": "No ID provided"}), 400

            conn = get_db_connection()
            cursor = conn.cursor()
            cursor.execute("DELETE FROM speech_to_text WHERE id = %s", (text_id,))
            conn.commit()
            cursor.close()
            conn.close()

            return jsonify({"message": "Record deleted successfully"})

        except Exception as e:
            return jsonify({"error": str(e)}), 500

# Run the Flask app
if __name__ == '__main__':
    app_instance = App(__name__)  # Create an instance of the App class
    app_instance.run(debug=True)

# Expose the Flask app for WSGI servers like Gunicorn
app = App(__name__).app