"""
modules/weather/fetch.py
━━━━━━━━━━━━━━━━━━━━━━━
WHAT TO ADD:
1. .env मध्ये WEATHER_API_KEY add करा
   Get from: https://openweathermap.org → My API Keys

HOW TO USE:
   from modules.weather.fetch import get_weather
   result = get_weather("Pune")
"""

import requests
from config import WEATHER_API_KEY

def get_weather(city: str) -> dict:
    """
    OpenWeatherMap API वरून weather fetch करतो

    Input:
        city (str) → e.g. "Pune", "Nashik", "Nagpur"

    Output:
        dict → {temperature, humidity, rain_forecast,
                farming_advice, description}
    """
    # ADD: WEATHER_API_KEY .env मध्ये असणे आवश्यक आहे
    if not WEATHER_API_KEY:
        return {
            "temperature":    25,
            "humidity":       60,
            "rain_forecast":  "No",
            "farming_advice": "Weather API key not set. Add to .env file.",
            "description":    "Unknown"
        }

    try:
        url = (
            f"http://api.openweathermap.org/data/2.5/forecast"
            f"?q={city}&appid={WEATHER_API_KEY}&units=metric"
        )
        res  = requests.get(url, timeout=5)
        data = res.json()

        if data.get("cod") != "200":
            raise ValueError(f"API Error: {data.get('message')}")

        item     = data["list"][0]
        temp     = item["main"]["temp"]
        humidity = item["main"]["humidity"]
        desc     = item["weather"][0]["description"]
        rain     = "Yes" if "rain" in item else "No"

        # ── Farming Advice ──────────────────────
        if rain == "Yes":
            advice = "Rain expected — skip irrigation and spraying today."
        elif humidity > 80:
            advice = "High humidity — fungal disease risk. Monitor crops closely."
        elif temp > 38:
            advice = "Very high temperature — irrigate early morning or evening."
        elif temp > 35:
            advice = "High temperature — water crops in early morning."
        elif temp < 10:
            advice = "Cold weather — risk of frost. Cover sensitive crops."
        else:
            advice = "Good farming conditions today."

        return {
            "temperature":    round(temp, 1),
            "humidity":       humidity,
            "rain_forecast":  rain,
            "farming_advice": advice,
            "description":    desc
        }

    except Exception as e:
        print(f"Weather API Error: {e}")
        return {
            "temperature":    25,
            "humidity":       60,
            "rain_forecast":  "No",
            "farming_advice": "Could not fetch weather. Check internet and API key.",
            "description":    "Unknown"
        }
print(get_weather("Pune"))
print(get_weather("Nagpur"))
print(get_weather("Sangli"))
print(get_weather("Kolhapur"))

