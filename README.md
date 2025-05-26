# AI-Project


```python
import requests
from autogen import ConversableAgent, UserProxyAgent

# Weatherstack API configuration
WEATHER_API_KEY = "2f9740e388f02f65f7f3715a7d6d9c21"
WEATHER_API_URL = "http://api.weatherstack.com/current"

# LLM configuration for Ollama
llm_config = {
    "model": "deepseek-r1:1.5b",
    "base_url": "http://localhost:11434/v1",
    "api_key": "ollama",
    "timeout": 120,
}

# WeatherAgent definition
weather_agent = ConversableAgent(
    name="WeatherAgent",
    llm_config=llm_config,
    system_message="You are WeatherAgent, an assistant that gives current weather information for any city.",
)

# UserAgent definition
user = UserProxyAgent(
    name="User",
    human_input_mode="ALWAYS",
    max_consecutive_auto_reply=0,
    code_execution_config={"use_docker": False},
    is_termination_msg=lambda msg: "exit" in msg.get("content", "").lower(),
)

# Function to get weather info from Weatherstack
def get_weather(city):
    params = {
        "access_key": WEATHER_API_KEY,
        "query": city,
        "units": "m"
    }
    try:
        response = requests.get(WEATHER_API_URL, params=params)
        data = response.json()
        if "current" in data:
            desc = data["current"]["weather_descriptions"][0]
            temp = data["current"]["temperature"]
            humidity = data["current"]["humidity"]
            wind_speed = data["current"]["wind_speed"]
            return f"🌤️ Weather in {city}:\n- Description: {desc}\n- Temp: {temp}°C\n- Humidity: {humidity}%\n- Wind Speed: {wind_speed} km/h"
        else:
            return f"⚠️ Could not retrieve weather for '{city}': {data.get('error', {}).get('info', 'Unknown error')}."
    except Exception as e:
        return f"❌ Error fetching weather: {str(e)}"

# Main loop
if __name__ == "__main__":
    print("🌦️ Start chatting with WeatherAgent! Type a city name or 'exit' to quit.\n")

    while True:
        user_msg = input("🌍 City name: ")
        if user_msg.strip().lower() == "exit":
            print("👋 Chat ended.")
            break

        weather_report = get_weather(user_msg.strip())
        user.initiate_chat(weather_agent, message=weather_report)


```
