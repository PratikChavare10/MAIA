"""
modules/irrigation/calculator.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
WHAT TO ADD:
- No training needed — formula based
- ADD: More crop stages in Kc_map if needed

HOW TO USE:
   from modules.irrigation.calculator import calculate_irrigation
   result = calculate_irrigation(
       weather={"rain_forecast": "No"},
       soil_moisture=35,
       crop_stage="flowering"
   )
"""

# ── Crop Coefficient Table ────────────────────────
# ADD: More crops/stages as needed
Kc_MAP = {
    "initial":    0.60,
    "vegetative": 0.85,
    "flowering":  1.15,
    "ripening":   0.90,
    "harvest":    0.70,
}

def calculate_irrigation(weather: dict,
                           soil_moisture: float,
                           crop_stage: str) -> dict:
    """
    ETc formula वापरून irrigation decision घेतो

    Input:
        weather       → dict from get_weather()
                        must have 'rain_forecast' key
        soil_moisture → 0-100 percentage
        crop_stage    → 'initial'/'vegetative'/
                        'flowering'/'ripening'/'harvest'

    Output:
        dict → {irrigate, decision, duration,
                water_need, method}
    """
    Kc  = Kc_MAP.get(crop_stage.lower(), 1.0)
    ET0 = 5.0          # mm/day (standard reference)
    ETc = round(Kc * ET0, 2)

    rain_coming = weather.get("rain_forecast", "No") == "Yes"
    humidity    = weather.get("humidity", 50)

    # ── Decision Logic ────────────────────────────
    if rain_coming:
        decision = "Skip — rain expected. Save water."
        duration = 0
        method   = "No irrigation"
    elif soil_moisture < 30:
        decision = f"Irrigate NOW — soil very dry ({soil_moisture}%)"
        duration = 60
        method   = "Drip irrigation recommended"
    elif soil_moisture < 50:
        decision = f"Irrigate today — {ETc}mm needed"
        duration = 45
        method   = "Drip or sprinkler irrigation"
    elif soil_moisture < 70:
        decision = "Irrigate tomorrow morning — soil adequate today"
        duration = 30
        method   = "Light irrigation"
    else:
        decision = "No irrigation needed — soil moisture sufficient"
        duration = 0
        method   = "No irrigation"

    # High humidity warning
    if humidity > 80 and not rain_coming:
        decision += " | High humidity — fungal risk, avoid wettin leaves"

    return {
        "irrigate":   duration > 0,
        "decision":   decision,
        "duration":   f"{duration} minutes",
        "water_need": f"{ETc} mm/day",
        "method":     method,
        "ETc":        ETc
    }
