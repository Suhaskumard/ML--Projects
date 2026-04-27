import re

def extract_features(text):
    text=text.lower()
    return {
        'has_email': int('@' in text),
        'has_salary': int('salary' in text or '$' in text),
        'urgent_words': int(any(w in text for w in ['urgent','quick money','immediate join'])),
        'link_count': len(re.findall(r'http', text))
    }

