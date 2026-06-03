import numpy as np

def setup_client():
    """Returns the proxy URL. Replaces the old InferenceClient setup."""
    # Verify the proxy is alive
    PROXY_URL = "https://cryotud-nb4170-llm-proxy.hf.space"
    import requests
    try:
        r = requests.get(PROXY_URL, timeout=10)
        r.raise_for_status()
        print("✅ Connected to NB4170 LLM proxy")
    except Exception as e:
        print(f"❌ Could not reach proxy: {e}")
        print("   Ask your instructor — the proxy may be sleeping.")
    return PROXY_URL

# DEFINING HELPER FUNCTIONS
def hex_to_rgb(hex_code):
    """Converts a 6-digit hex string (for example: 'FF0000') to an RGB tuple"""
    r = int(hex_code[0:2], 16)
    g = int(hex_code[2:4], 16)
    b = int(hex_code[4:6], 16)
    return r, g, b

def colored(text, bg_hex):
    """
    Applies the given background hex color to text (using 'ANSI 24-bit')
    A black foreground (000000) is used for contrast
    """
    fg_r, fg_g, fg_b = 0, 0, 0  # Black foreground
    bg_r, bg_g, bg_b = hex_to_rgb(bg_hex)

    # ANSI escape code: (1) means bold and (48;2) background and (38;2) foreground
    # create formatted string and apply the background color to the text
    return f"\033[1;38;2;{fg_r};{fg_g};{fg_b};48;2;{bg_r};{bg_g};{bg_b}m{text}\033[0m"

def prob_to_color(avg_prob):
    """
    Maps the avg standard probability (a value between 0.0 and 1.0) of the sentence to a smooth color 
    gradient from red (0.0) to green (1.0)
    """
    # Clamp the input probability between 0.0 and 1.0 to avoid pointing float errors
    norm_prob = np.clip(avg_prob, 0.0, 1.0)
    
    # Red value decreases as probability increases
    r = int((1 - norm_prob) * 255)
    # Green value increases as probability increases 
    g = int(norm_prob * 255)
    # Blue value remains 0
    b = 0
    
    return f"{r:02x}{g:02x}{b:02x}"

def query_model(prompt, model_name, max_tokens, client, system_prompt=""):
    """Send one prompt to the proxy and return the model's response.

    Parameters
    ----------
    prompt        : str   the user prompt (the question or batch of questions)
    model_name    : str   "llama-1b", or "llama-8b" 
    max_tokens    : int   how many tokens to generate
    client        : str   proxy URL from setup_client()
    system_prompt : str   optional system instruction

    Returns
    -------
    dict with keys:
        "answer"      : str   the generated text
        "logprobs"    : dict  {token: logprob}
        "token_probs" : dict  {token: probability}
        "logprob_contents" : list of dictionaries for each generated token[{}]
    or None if the request was rate-limited / rejected.
    """
    import requests 
    response = requests.post(
        f"{client}/generate",
        json={
            "system_prompt": system_prompt,
            "prompt": prompt,
            "model": model_name,
            "max_tokens": max_tokens,
        },
        timeout=60,
    )

    if response.status_code == 429:
        print("Rate limit hit - wait a moment and try again.")
        return None
    if response.status_code == 400:
        print(f"Bad request: {response.json().get('detail')}")
        return None

    response.raise_for_status()
    return response.json()