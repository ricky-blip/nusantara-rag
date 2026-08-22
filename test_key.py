from openai import OpenAI
import os
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_BASE_URL"),
)

resp = client.chat.completions.create(
    model=os.getenv("CHAT_MODEL", "notispace-v1"),
    messages=[{"role": "user", "content": "Halo, balas dengan: KONEKSI BERHASIL"}],
)
print("Jawaban model:", resp.choices[0].message.content)