import numpy as np

def setup_client():
    """Returns the proxy URL. Replaces the old InferenceClient setup."""
    # Verify the proxy is alive
    PROXY_URL = "https://cryotud-nb4170-llm-proxy.hf.space"
    import requests
    try:
        r = requests.get(PROXY_URL, timeout=10)
        r.raise_for_status()
        print("✅ Connected to TUDelft LLM proxy")
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