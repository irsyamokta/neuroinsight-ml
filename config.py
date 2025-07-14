from dotenv import load_dotenv
import os

load_dotenv()

def get_allowed_origins():
    origins = os.getenv("ALLOWED_ORIGINS", "")
    return [origin.strip() for origin in origins.split(",") if origin.strip()]