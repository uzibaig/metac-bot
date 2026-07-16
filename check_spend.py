"""
One-command OpenRouter spend check for the forecasting bots.

    poetry run python check_spend.py

Reads the balance straight from OpenRouter (the billing ground truth — the
bot's own logged "estimated cost" lines are unreliable: litellm's price table
overcounts gpt-4o-search-preview and can't see perplexity/sonar at all).
"""

import os

import dotenv
import requests

dotenv.load_dotenv()

API_KEY = os.environ["OPENROUTER_API_KEY"]
HEADERS = {"Authorization": f"Bearer {API_KEY}"}

credits = requests.get(
    "https://openrouter.ai/api/v1/credits", headers=HEADERS, timeout=30
).json()["data"]
key_info = requests.get(
    "https://openrouter.ai/api/v1/auth/key", headers=HEADERS, timeout=30
).json()["data"]

total = credits["total_credits"]
used = credits["total_usage"]
remaining = total - used

print(f"OpenRouter credits:  ${total:.2f}")
print(f"Used so far:         ${used:.2f}")
print(f"Remaining:           ${remaining:.2f}")
if key_info.get("limit") is not None:
    print(f"Key spend limit:     ${key_info['limit']:.2f}")
else:
    print("Key spend limit:     none set  <- consider setting one at openrouter.ai/keys")

print()
print(f"~ {int(remaining / 0.33)} more v2 shadow questions at ~$0.33 each")
print(f"~ {int(remaining / 0.05)} more template forecasts at ~$0.05 each")
