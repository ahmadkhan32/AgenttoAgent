import os
import asyncio
from google import genai
from google.genai import types

async def test_key():
    client = genai.Client(api_key=os.environ.get("GOOGLE_API_KEY", "AIzaSyB9J-lMWTNflYCWbxU5UHohFnoiKhOlAag"))
    try:
        response = client.models.generate_content(
            model='gemini-1.5-flash',
            contents='Say hi briefly.'
        )
        print("Success gemini-2.0-flash:", response.text)
    except Exception as e:
        print("Error gemini-2.0-flash:", str(e))

if __name__ == "__main__":
    asyncio.run(test_key())
