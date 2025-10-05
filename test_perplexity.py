import os
import requests
import re
from dotenv import load_dotenv

load_dotenv()
PERPLEXITY_API_KEY = os.getenv('PERPLEXITY_API_KEY')
PERPLEXITY_API_URL = 'https://api.perplexity.ai/chat/completions'

year = 2024
prompt = f'Give me the URLs to the Qantas {year} AGM notice of meeting, AGM results, and AGM minutes/transcript. Return only the direct PDF URLs.'

headers = {
    'Authorization': f'Bearer {PERPLEXITY_API_KEY}',
    'Content-Type': 'application/json'
}

payload = {
    'model': 'sonar',
    'messages': [{'role': 'user', 'content': prompt}],
    'temperature': 0.2,
    'max_tokens': 1000
}

print(f"Testing Perplexity API for {year} AGM documents...")
response = requests.post(PERPLEXITY_API_URL, json=payload, headers=headers, timeout=30)
result = response.json()
answer = result['choices'][0]['message']['content']
print('\nResponse:', answer)
print()

url_pattern = r'https?://[^\s<>"{}|\\^`\[\]]+\.pdf'
found_urls = re.findall(url_pattern, answer)
print('Found URLs:', found_urls)
