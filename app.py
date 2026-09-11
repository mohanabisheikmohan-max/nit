import os
from pathlib import Path

from dotenv import load_dotenv
from flask import Flask, jsonify, render_template, request
from google import genai
from google.genai import types

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent
MODEL_NAME = "gemini-3.1-flash-lite"
CONFIG_FILE = BASE_DIR / "chatbot_config"

app = Flask(__name__, template_folder="templates")

api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    raise RuntimeError("GEMINI_API_KEY is not configured.")

client = genai.Client(api_key=api_key)
system_prompt = CONFIG_FILE.read_text(encoding="utf-8").strip()


@app.get("/")
def index():
    return render_template("index.html")


@app.post("/api/chat")
def chat():
    data = request.get_json(silent=True) or {}
    message = str(data.get("message", "")).strip()

    if not message:
        return jsonify({"error": "Please enter a message."}), 400

    try:
        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=message,
            config=types.GenerateContentConfig(
                system_instruction=system_prompt,
                temperature=0.2,
                max_output_tokens=1024,
            ),
        )

        reply = (response.text or "").strip()
        return jsonify({"reply": reply or "I couldn't generate a response right now."})

    except Exception:
        app.logger.exception("Gemini request failed")
        return jsonify({
            "error": "I couldn't process that request right now. Please try again."
        }), 500


if __name__ == "__main__":
    app.run()
