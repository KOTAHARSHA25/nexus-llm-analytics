import os
import sys
from dotenv import load_dotenv

sys.path.append('src')
load_dotenv()

from backend.core.online_clients import GroqClient, OpenRouterClient, CohereEmbedClient, HuggingFaceEmbedClient, FirecrawlClient, GeminiClient

def test_apis():
    print('--- Testing Groq ---')
    try:
        groq = GroqClient(os.getenv('GROQ_API_KEY'))
        res = groq.generate('List the numbers 1 to 3', tier='simple')
        print('Groq Success:', res[:50].replace('\n', ' '))
    except Exception as e:
        print('Groq Error:', e)

    print('\n--- Testing Gemini ---')
    try:
        gemini = GeminiClient(os.getenv('GEMINI_API_KEY'))
        res = gemini.generate('List the numbers 1 to 3', tier='simple')
        print('Gemini Success:', res[:50].replace('\n', ' '))
    except Exception as e:
        print('Gemini Error:', e)
        
    print('\n--- Testing OpenRouter ---')
    try:
        or_client = OpenRouterClient(os.getenv('OPENROUTER_API_KEY'))
        res = or_client.chat([{'role':'user', 'content':'What is 2+2?'}], tier='simple')
        print('OpenRouter Success:', res[:50].replace('\n', ' '))
    except Exception as e:
        print('OpenRouter Error:', e)

    print('\n--- Testing Cohere ---')
    try:
        cohere = CohereEmbedClient(os.getenv('COHERE_API_KEY'))
        res = cohere.embed_query('Hello world')
        print('Cohere Success: Embedding vector length', len(res))
    except Exception as e:
        print('Cohere Error:', e)

    print('\n--- Testing HuggingFace ---')
    try:
        hf = HuggingFaceEmbedClient(os.getenv('HUGGINGFACE_API_KEY'))
        res = hf.embed_query('Hello world')
        print('HuggingFace Success: Embedding vector length', len(res))
    except Exception as e:
        print('HuggingFace Error:', e)

    print('\n--- Testing Firecrawl ---')
    try:
        fc = FirecrawlClient(os.getenv('FIRECRAWL_API_KEY'))
        res = fc.scrape('https://example.com')
        print('Firecrawl scrape Success: length', len(str(res)))
    except Exception as e:
        print('Firecrawl Error:', e)

if __name__ == '__main__':
    test_apis()
