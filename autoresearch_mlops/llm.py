import requests

def suggest(history, program):
    prompt = f"""
{program}

History:
{history}

Return JSON config.
"""

    r = requests.post(
        "http://localhost:11434/api/generate",
        json={
            "model": "gemma:2b",
            "prompt": prompt,
            "stream": False
        }
    )

    try:
        return eval(r.json()["response"])
    except:
        return {}
