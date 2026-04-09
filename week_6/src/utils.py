# def setup_client():
#     import os 
#     import json 
#     from huggingface_hub import InferenceClient

#     # ACCESS_TOKEN_PATH="/Users/alokbharadwaj/Desktop/hf_access.json"
#     # with open(ACCESS_TOKEN_PATH, "r") as f:
#     #     access_token = json.load(f)["token"]
#     access_token = "hf_xxxxxxxxxxxxxx"

#     client = InferenceClient(token=access_token)
#     return client


def setup_client():
    """Returns the proxy URL. Replaces the old InferenceClient setup."""
    # Verify the proxy is alive
    PROXY_URL = "https://alokbharadwaj-llm-uncertainty.hf.space"
    import requests
    try:
        r = requests.get(PROXY_URL, timeout=10)
        r.raise_for_status()
        print("✅ Connected to TUDelft LLM proxy")
    except Exception as e:
        print(f"❌ Could not reach proxy: {e}")
        print("   Ask your instructor — the proxy may be sleeping.")
    return PROXY_URL