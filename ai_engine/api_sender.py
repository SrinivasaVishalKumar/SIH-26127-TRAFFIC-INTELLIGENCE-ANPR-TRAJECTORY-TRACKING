import json
import requests
import os
import time

VERCEL_URL = "https://sih-127-vercel.vercel.app/api/add-log"

# Safely points to the 'database' folder exactly one level up from this script
FILE_PATH = os.path.join(os.path.dirname(__file__), "..", "database", "log.json")

print("Starting continuous API sender...")

while True:
    try:
        with open(FILE_PATH, "r") as file:
            logs = json.load(file)

        if len(logs) > 0:
            print(f"Loaded {len(logs)} logs. Sending to Vercel...\n")
            for log in logs:
                try:
                    response = requests.post(
                        VERCEL_URL, json=log, headers={"Content-Type": "application/json"}
                    )
                    print(f"Status: {response.status_code} | Body: {response.text.strip()}")
                except Exception as e:
                    print(f"[Request Error]: {e}")
            
            # Clear the file after sending so it doesn't resend duplicates
            with open(FILE_PATH, "w") as file:
                json.dump([], file)
            print("-" * 50)
            
    except FileNotFoundError:
        print(f"Waiting... {FILE_PATH} not found.")
    except json.JSONDecodeError:
        pass # Ignore errors if the file is currently being written to by the tracker

    # Wait 15 seconds before checking the file again
    time.sleep(15)