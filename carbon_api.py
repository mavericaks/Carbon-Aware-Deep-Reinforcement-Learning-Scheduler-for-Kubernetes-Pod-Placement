import time
import math
import random
from datetime import datetime, timezone
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel

app = FastAPI(title="Mock Carbon Intensity API")

# Helper function to get carbon intensity based on current UTC time
def calculate_carbon_intensity(zone: str) -> float:
    now = datetime.now(timezone.utc)
    hour = now.hour + now.minute / 60.0 + now.second / 3600.0

    if zone == "us-east":
        # Stable coal-heavy intensity
        base = 550.0
        fluctuation = random.uniform(-15.0, 15.0)
        return max(400.0, base + fluctuation)
    elif zone == "eu-west":
        # Wind-heavy, fluctuating across hours
        base = 250.0
        # Wind cycle repeats every 4 hours
        wind_cycle = 80.0 * math.sin(hour / 4.0 * 2 * math.pi)
        fluctuation = random.uniform(-20.0, 20.0)
        return max(80.0, base + wind_cycle + fluctuation)
    elif zone == "us-west":
        # Solar diurnal curve: lowest carbon during midday (10:00 to 16:00 UTC/local)
        base = 300.0
        # Diurnal cycle repeats every 24 hours
        solar_dip = -200.0 * max(0.0, math.sin(hour / 24.0 * 2 * math.pi))
        fluctuation = random.uniform(-10.0, 10.0)
        return max(40.0, base + solar_dip + fluctuation)
    else:
        raise HTTPException(status_code=404, detail=f"Zone '{zone}' not found. Supported zones: us-east, eu-west, us-west")

class CarbonIntensityResponse(BaseModel):
    zone: str
    carbonIntensity: float
    unit: str = "gCO2eq/kWh"
    datetime: str
    updatedAt: str
    isEstimated: bool = False

@app.get("/latest", response_model=CarbonIntensityResponse)
def get_latest(zone: str = Query(..., description="Electricity zone, e.g. us-east, eu-west, us-west")):
    intensity = calculate_carbon_intensity(zone)
    now_str = datetime.now(timezone.utc).isoformat()
    return CarbonIntensityResponse(
        zone=zone,
        carbonIntensity=round(intensity, 2),
        datetime=now_str,
        updatedAt=now_str
    )

@app.get("/forecast")
def get_forecast(zone: str = Query(..., description="Electricity zone, e.g. us-east, eu-west, us-west")):
    # Return 24-hour forecast
    forecasts = []
    now = datetime.now(timezone.utc)
    
    for h in range(24):
        future_time = now.hour + h
        # Project future time
        if zone == "us-east":
            intensity = 550.0 + random.uniform(-10.0, 10.0)
        elif zone == "eu-west":
            intensity = 250.0 + 80.0 * math.sin((future_time) / 4.0 * 2 * math.pi)
        elif zone == "us-west":
            intensity = 300.0 - 200.0 * max(0.0, math.sin((future_time) / 24.0 * 2 * math.pi))
        else:
            raise HTTPException(status_code=404, detail="Zone not found")
        
        forecasts.append({
            "carbonIntensity": round(intensity, 2),
            "datetime": (now.replace(hour=(now.hour + h) % 24)).isoformat()
        })
    
    return {
        "zone": zone,
        "forecast": forecasts
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
