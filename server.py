import os
import base64
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
from groq import Groq
from google import genai
from google.genai import types

app = FastAPI()

# Master keys are fetched securely from server environment variables
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")

class AIRequest(BaseModel):
    mode: str  # "code" or "mcq"
    image_base64: str

@app.post("/process-ai")
def process_ai(data: AIRequest):
    try:
        img_bytes = base64.b64decode(data.image_base64)

        if data.mode == "code":
            if not GROQ_API_KEY:
                raise HTTPException(status_code=500, detail="Groq API key not configured on server.")
            
            groq_client = Groq(api_key=GROQ_API_KEY)
            base64_image = data.image_base64
            
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Read the coding problem visible on this screenshot. Provide ONLY the exact, optimized Python code to solve the problem. Do NOT use markdown code blocks. Return the raw indented code directly."},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}},
                    ],
                }
            ]

            stream_response = groq_client.chat.completions.create(
                model="qwen/qwen3.6-27b", 
                messages=messages, 
                temperature=0.3
            )
            result_text = stream_response.choices[0].message.content
            return {"result": result_text}

        elif data.mode == "mcq":
            if not GEMINI_API_KEY:
                raise HTTPException(status_code=500, detail="Gemini API key not configured on server.")
            
            gemini_client = genai.Client(api_key=GEMINI_API_KEY)
            response = gemini_client.models.generate_content(
                model="gemini-3.6-flash",
                contents=[
                    types.Part.from_bytes(data=img_bytes, mime_type='image/jpeg'),
                    "Solve this multiple-choice question. Output EXACTLY ONE LINE with just the correct option label and choice text (e.g., 'Option B'). No explanation."
                ]
            )
            result_text = response.text.strip() if response.text else "Option ?"
            return {"result": result_text}
            
        else:
            raise HTTPException(status_code=400, detail="Invalid mode specified.")

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
