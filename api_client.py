"""
MAIA — thin wrapper around the FastAPI backend for the Streamlit frontend.
"""

import json
import requests

BASE_URL = "http://localhost:8000"


def register(name, email, password, city, language):
    r = requests.post(f"{BASE_URL}/api/auth/register", data={
        "name": name, "email": email, "password": password,
        "city": city, "language": language,
    })
    return r.json()


def login(email, password):
    r = requests.post(f"{BASE_URL}/api/auth/login", data={"email": email, "password": password})
    if r.status_code != 200:
        return {"ok": False, "msg": "Invalid email or password."}
    return r.json()


def _headers(token):
    return {"Authorization": f"Bearer {token}"}


def get_threads(token):
    r = requests.get(f"{BASE_URL}/api/threads", headers=_headers(token))
    return r.json().get("threads", [])


def get_thread_messages(token, thread_id):
    r = requests.get(f"{BASE_URL}/api/threads/{thread_id}", headers=_headers(token))
    return r.json().get("messages", [])


def new_thread(token, mode):
    r = requests.post(f"{BASE_URL}/api/threads/new", headers=_headers(token), json={"mode": mode})
    return r.json().get("thread_id")


def delete_thread(token, thread_id):
    r = requests.delete(f"{BASE_URL}/api/threads/{thread_id}", headers=_headers(token))
    return r.json()


def get_weather(token, city):
    r = requests.get(f"{BASE_URL}/api/weather", headers=_headers(token), params={"city": city})
    return r.json()


def stream_chat(token, result_holder=None, **form_data):
    files = form_data.pop("files", None)
    with requests.post(
        f"{BASE_URL}/api/chat/stream",
        headers=_headers(token),
        data=form_data,
        files=files,
        stream=True,
        timeout=920,
    ) as r:
        for line in r.iter_lines(decode_unicode=True):
            if not line or not line.startswith("data:"):
                continue
            payload = json.loads(line[len("data:"):].strip())
            if "token" in payload:
                yield payload["token"]
            elif "error" in payload:
                yield f"\n\n⚠️ Error: {payload['error']}"
            elif payload.get("done"):
                if result_holder is not None:
                    result_holder["thread_id"] = payload.get("thread_id")
                return