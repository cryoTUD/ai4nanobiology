def setup_client():
    import os 
    import json 
    from huggingface_hub import InferenceClient
    
    ACCESS_TOKEN_PATH="/Users/alokbharadwaj/Desktop/hf_access.json"
    with open(ACCESS_TOKEN_PATH, "r") as f:
        access_token = json.load(f)["token"]

    client = InferenceClient(token=access_token)
    return client
