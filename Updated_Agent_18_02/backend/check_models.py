import os
from dotenv import load_dotenv
import google.generativeai as genai

load_dotenv()
api_key = os.getenv("ABHAY_API_KEY")

if not api_key:
    print("No API key found in .env (ABHAY_API_KEY)")
else:
    print(f"Using API Key: {api_key[:5]}...")
    genai.configure(api_key=api_key)

    print("\nListing available models that support generateContent:")
    try:
        for m in genai.list_models():
            if 'generateContent' in m.supported_generation_methods:
                print(f"- {m.name}")
    except Exception as e:
        print(f"Error listing models: {e}")
