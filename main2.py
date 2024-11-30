from flask import Flask, request, jsonify, send_file, send_from_directory
from werkzeug.utils import secure_filename
import os
from gtts import gTTS
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import speech_recognition as sr
import fitz  # PyMuPDF for PDF text extraction
from docx import Document

# Constants
ALLOWED_EXTENSIONS = {'txt', 'pdf', 'docx'}
TRANSCRIBED_TEXT_FILE = "transcribed_text.txt"

class DocumentComparison:
    @staticmethod
    def allowed_file(filename):
        """Check if the file has an allowed extension"""
        return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

    @staticmethod
    def preprocess(text):
        """Preprocess text for comparison (convert to lowercase)"""
        return text.lower()

    @staticmethod
    def compare(doc1, doc2):
        """Compare two documents based on text similarity"""
        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform([doc1, doc2])
        cosine_sim = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])
        similarity_percentage = cosine_sim[0][0] * 100
        return similarity_percentage

class TextToSpeech:
    @staticmethod
    def convert_to_speech(text, output_folder, filename):
        """Convert text to speech and save as an MP3 file"""
        try:
            tts = gTTS(text=text, lang='si')  # Sinhala language
            output_file = os.path.join(output_folder, f'{filename}.mp3')
            tts.save(output_file)
            return output_file
        except Exception as e:
            print(f"Error generating MP3: {e}")
            return None

    @staticmethod
    def extract_text(file_path, filename):
        """Extract text based on file type"""
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

class SpeechRecognitionService:
    @staticmethod
    def record_and_convert_sinhala_speech():
        """Record and convert Sinhala speech to text"""
        recognizer = sr.Recognizer()
        try:
            # Capture audio from the microphone
            with sr.Microphone() as source:
                print("Adjusting for ambient noise...")
                recognizer.adjust_for_ambient_noise(source)
                print("Recording... Please speak.")
                audio_data = recognizer.listen(source)

            # Try recognizing the speech
            text = recognizer.recognize_google(audio_data, language="si-LK")
            print("Transcribed Text: ", text)

            # Save transcribed text to a file
            with open(TRANSCRIBED_TEXT_FILE, 'w', encoding='utf-8') as file:
                file.write(text)

            return {"success": True, "transcribed_text": text}, 200
        except sr.UnknownValueError:
            return {"success": False, "error": "Could not understand the audio"}, 400
        except sr.RequestError as e:
            return {"success": False, "error": f"Error with speech recognition service: {e}"}, 500
        except Exception as e:
            return {"success": False, "error": f"An error occurred: {e}"}, 500

class FileService:
    @staticmethod
    def save_file(file, upload_folder):
        """Save the uploaded file"""
        if not os.path.exists(upload_folder):
            os.makedirs(upload_folder)
        filename = secure_filename(file.filename)
        file_path = os.path.join(upload_folder, filename)
        file.save(file_path)
        return file_path

    @staticmethod
    def get_all_files(folder):
        """Retrieve all files in the folder"""
        if not os.path.exists(folder):
            return [], "Folder does not exist"
        files = [f for f in os.listdir(folder) if f.endswith('.mp3')]
        return files, "No files found" if not files else ""

# Flask Application
class App:
    def __init__(self, name):
        self.app = Flask(name)

        # Register routes
        self.app.add_url_rule('/compare', 'compare_documents', self.compare_documents, methods=['POST'])
        self.app.add_url_rule('/convert', 'convert_text_to_speech', self.convert_text_to_speech, methods=['POST'])
        self.app.add_url_rule('/record_and_convert_sinhala_speech', 'record_and_convert_sinhala_speech', self.record_and_convert_sinhala_speech, methods=['POST'])
        self.app.add_url_rule('/voice/<filename>', 'serve_voice', self.serve_voice, methods=['GET'])
        self.app.add_url_rule('/voices', 'get_all_voice_files', self.get_all_voice_files, methods=['GET'])
        self.app.add_url_rule('/transcribe', 'transcribe_audio', self.transcribe_audio, methods=['POST'])
        self.app.add_url_rule('/get_transcribed_text', 'get_transcribed_text', self.get_transcribed_text, methods=['GET'])
        self.app.add_url_rule('/delete_transcribed_text', 'delete_transcribed_text', self.delete_transcribed_text, methods=['DELETE'])
        self.app.add_url_rule('/update_transcribed_text', 'update_transcribed_text', self.update_transcribed_text, methods=['PUT'])

    def run(self, debug=True):
        self.app.run(debug=debug)

    # Document Comparison Endpoint
    def compare_documents(self):
        try:
            doc1_file = request.files.get('document1')
            doc2_file = request.files.get('document2')

            if not doc1_file or not doc2_file:
                return jsonify({'error': 'Both documents must be provided'}), 400

            doc1 = doc1_file.read().decode('utf-8')
            doc2 = doc2_file.read().decode('utf-8')

            doc1 = DocumentComparison.preprocess(doc1)
            doc2 = DocumentComparison.preprocess(doc2)

            similarity_percentage = DocumentComparison.compare(doc1, doc2)

            return jsonify({'similarity_percentage': similarity_percentage})

        except Exception as e:
            return jsonify({'error': str(e)}), 500

    # Text-to-Speech Endpoint
    def convert_text_to_speech(self):
        try:
            if 'file' not in request.files:
                return jsonify({'error': 'No file part'}), 400

            file = request.files['file']
            if file.filename == '':
                return jsonify({'error': 'No selected file'}), 400

            if not DocumentComparison.allowed_file(file.filename):
                return jsonify({'error': 'Unsupported file format'}), 400

            upload_folder = os.path.join(self.app.root_path, 'uploads')
            file_path = FileService.save_file(file, upload_folder)

            input_text = TextToSpeech.extract_text(file_path, file.filename)

            output_folder = os.path.join(self.app.root_path, 'voice_messages')
            if not os.path.exists(output_folder):
                os.makedirs(output_folder)

            base_filename = os.path.splitext(file.filename)[0]
            output_file_path = TextToSpeech.convert_to_speech(input_text, output_folder, base_filename)

            if output_file_path:
                return send_file(output_file_path, as_attachment=True)
            else:
                return jsonify({'error': 'Failed to generate MP3'}), 500

        except Exception as e:
            return jsonify({'error': str(e)}), 500

    # Sinhala Speech Recognition Endpoint
    def record_and_convert_sinhala_speech(self):
        response, status = SpeechRecognitionService.record_and_convert_sinhala_speech()
        return jsonify(response), status

    # Serve Voice File Endpoint
    def serve_voice(self, filename):
        try:
            voice_folder = os.path.join(self.app.root_path, 'voice_messages')
            return send_from_directory(voice_folder, filename)
        except Exception as e:
            return str(e), 500

    # Retrieve all voice files
    def get_all_voice_files(self):
        voice_folder = os.path.join(self.app.root_path, 'voice_messages')
        voice_files, error_msg = FileService.get_all_files(voice_folder)
        if not voice_files:
            return jsonify({"success": False, "error": error_msg}), 404
        return jsonify({"success": True, "voice_files": voice_files}), 200

    # Alternative Speech to Text Routes (File Upload Based)
    def transcribe_audio(self):
        try:
            if 'audio' not in request.files:
                return jsonify({"success": False, "error": "No audio file provided"}), 400

            audio_file = request.files['audio']
            if audio_file.filename == '':
                return jsonify({"success": False, "error": "No selected file"}), 400

            audio_folder = os.path.join(self.app.root_path, 'audio_uploads')
            if not os.path.exists(audio_folder):
                os.makedirs(audio_folder)

            audio_path = FileService.save_file(audio_file, audio_folder)

            with open(TRANSCRIBED_TEXT_FILE, 'w', encoding='utf-8') as file:
                file.write(f"Audio file received: {audio_file.filename}")

            return jsonify({
                "success": True,
                "message": "Audio file received",
                "filename": audio_file.filename
            }), 200
        except Exception as e:
            return jsonify({"success": False, "error": str(e)}), 500

    def get_transcribed_text(self):
        if os.path.exists(TRANSCRIBED_TEXT_FILE):
            with open(TRANSCRIBED_TEXT_FILE, 'r', encoding='utf-8') as file:
                transcribed_text = file.read()
            return jsonify({"success": True, "transcribed_text": transcribed_text}), 200
        else:
            return jsonify({"success": False, "error": "Transcribed text file not found."}), 404

    def delete_transcribed_text(self):
        if os.path.exists(TRANSCRIBED_TEXT_FILE):
            os.remove(TRANSCRIBED_TEXT_FILE)
            return jsonify({"success": True, "message": "Transcribed text file deleted."}), 200
        else:
            return jsonify({"success": False, "error": "Transcribed text file not found."}), 404

    def update_transcribed_text(self):
        if os.path.exists(TRANSCRIBED_TEXT_FILE):
            new_text = request.json.get('new_text')
            if new_text:
                with open(TRANSCRIBED_TEXT_FILE, 'w', encoding='utf-8') as file:
                    file.write(new_text)
                return jsonify({"success": True, "message": "Transcribed text updated.", "new_text": new_text}), 200
            else:
                return jsonify({"success": False, "error": "No new text provided."}), 400
        else:
            return jsonify({"success": False, "error": "Transcribed text file not found."}), 404


if __name__ == "__main__":
    app = App(__name__)
    app.run(debug=True)
