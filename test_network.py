import requests

print("Testing network access...")
response = requests.get("https://www.google.com", timeout=10)
print("Success! Status code:", response.status_code)