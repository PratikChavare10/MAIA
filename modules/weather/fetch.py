from datetime import date, timedelta
import pandas as pd
import requests
from config import WEATHER_API_KEY


def get_weather(city: str) -> dict:
    """Combines OpenWeatherMap forecast advice/rain status with

    Open-Meteo 12-month aggregated historical metrics.
    """
    result = {
        "city": city,
        "temperature": 25.0,  
        "humidity": 60.0,  
        "rainfall": 0.0,  
        "rain_forecast": "No",  
        "description": "Unknown",  
        "farming_advice": "No advice generated",
    }

    # ==========================================
    # PART 1: Open-Meteo Historical Data
    # ==========================================
    try:
        # Step 1.1: Geocoding
        geo_url = "https://geocoding-api.open-meteo.com/v1/search"
        geo_params = {
            "name": city,
            "count": 1,
            "language": "en",
            "format": "json",
        }
        geo_res = requests.get(geo_url, params=geo_params, timeout=5).json()

        if geo_res.get("results"):
            location = geo_res["results"][0]
            lat, lon = location["latitude"], location["longitude"]
            result["city"] = (
                f"{location['name']}, {location.get('country', '')}"
            )

            # Step 1.2: Archive dates setup
            end_date = date.today() - timedelta(days=2)
            start_date = end_date - timedelta(days=365)

            # Step 1.3: Historical Archive Request
            archive_url = "https://archive-api.open-meteo.com/v1/archive"
            archive_params = {
                "latitude": lat,
                "longitude": lon,
                "start_date": start_date.strftime("%Y-%m-%d"),
                "end_date": end_date.strftime("%Y-%m-%d"),
                "daily": [
                    "temperature_2m_mean",
                    "relative_humidity_2m_mean",
                    "precipitation_sum",
                ],
                "timezone": "auto",
            }
            archive_res = requests.get(
                archive_url, params=archive_params, timeout=5
            ).json()
            daily_data = archive_res.get("daily", {})

            if daily_data:
                # Step 1.4: Resampling & Monthly Averages calculation
                df = pd.DataFrame(
                    {
                        "date": pd.to_datetime(daily_data["time"]),
                        "temperature": daily_data["temperature_2m_mean"],
                        "humidity": daily_data["relative_humidity_2m_mean"],
                        "rainfall": daily_data["precipitation_sum"],
                    }
                )
                df.set_index("date", inplace=True)

                monthly_df = df.resample("MS").agg(
                    {"rainfall": "sum", "temperature": "mean", "humidity": "mean"}
                )

                # Extracts the requested calculations:
                result["temperature"] = round(
                    float(monthly_df["temperature"].sum() / 12), 2
                )
                result["humidity"] = round(
                    float(monthly_df["humidity"].sum() / 12), 2
                )
                result["rainfall"] = round(
                    float(monthly_df["rainfall"].sum() / 12), 2
                )

    except Exception as e:
        print(f"Open-Meteo Historical API Error: {e}")

    # ==========================================
    # PART 2: OpenWeatherMap Forecast Data
    # ==========================================
    if not WEATHER_API_KEY:
        result["farming_advice"] = (
            "Weather API key not set. Add WEATHER_API_KEY to config."
        )
        return result

    try:
        url = (
            f"http://api.openweathermap.org/data/2.5/forecast"
            f"?q={city}&appid={WEATHER_API_KEY}&units=metric"
        )
        res = requests.get(url, timeout=5)
        data = res.json()

        if data.get("cod") == "200" and data.get("list"):
            item = data["list"][0]
            current_temp = item["main"]["temp"]
            current_humidity = item["main"]["humidity"]
            desc = item["weather"][0]["description"]
            rain = "Yes" if "rain" in item else "No"

            # Assign rain and description
            result["rain_forecast"] = rain
            result["description"] = desc

            # Generate Farming Advice based on short-term forecast
            if rain == "Yes":
                advice = (
                    "Rain expected — skip irrigation and spraying today."
                )
            elif current_humidity > 80:
                advice = "High humidity — fungal disease risk. Monitor crops closely."
            elif current_temp > 38:
                advice = (
                    "Very high temperature — irrigate early morning or evening."
                )
            elif current_temp > 35:
                advice = "High temperature — water crops in early morning."
            elif current_temp < 10:
                advice = "Cold weather — risk of frost. Cover sensitive crops."
            else:
                advice = "Good farming conditions today."

            result["farming_advice"] = advice
        else:
            result["farming_advice"] = (
                f"API Error: {data.get('message', 'Failed to retrieve forecast')}"
            )

    except Exception as e:
        print(f"OpenWeather API Error: {e}")
        result["farming_advice"] = (
            "Could not fetch weather forecast. Check internet connection."
        )

    return result


# Example Execution
if __name__ == "__main__":
    weather_info = get_weather("Pune")
    print(weather_info)