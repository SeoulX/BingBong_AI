import google.generativeai as genai
import os

genai.configure(api_key='AIzaSyBDbmE9bwOEmSNMFUbIsKEF7SyL-3-kr5I')

generation_config = {
  "temperature": 1,
  "top_p": 0.95,
  "top_k": 64,
  "max_output_tokens": 8192,
  "response_mime_type": "text/plain",
}

model = genai.GenerativeModel(
  model_name="gemini-1.5-flash",
  generation_config=generation_config,
  # safety_settings = Adjust safety settings
  # See https://ai.google.dev/gemini-api/docs/safety-settings
)
response = model.generate_content('Teach me about how an LLM works')