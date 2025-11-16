import requests
import folium
from IPython.display import display, IFrame

# ========== CONFIGURATION ==========
OSRM_URL = "http://127.0.0.1:5000"
PROFILE = "driving"

# Gandhipuram → Thirumalayampalayam (Coimbatore)
src = (76.9663, 11.0171)   # (longitude, latitude)
dst = (76.8979, 10.8765)   # (longitude, latitude)

# ========== OSRM ROUTE FUNCTION ==========
def osrm_route(base_url, src, dst, profile="driving"):
    url = f"{base_url}/route/v1/{profile}/{src[0]},{src[1]};{dst[0]},{dst[1]}?overview=full&geometries=geojson"
    print(f"Requesting route: {url}")
    r = requests.get(url)
    if r.status_code == 200:
        data = r.json()
        if data.get("code") == "Ok":
            print("✅ Route fetched successfully!")
            return data
        else:
            print(f"⚠️ OSRM error: {data.get('message', 'Unknown error')}")
    else:
        print(f"❌ HTTP error: {r.status_code}")
    return None

# ========== PRETTY PRINT ==========
def pretty_route_summary(data):
    route = data["routes"][0]
    distance_km = route["distance"] / 1000
    duration_min = route["duration"] / 60
    return f"🛣 Distance: {distance_km:.2f} km | ⏱ Duration: {duration_min:.2f} min"

# ========== FOLIUM MAP CREATOR ==========
def route_to_folium(data):
    route = data["routes"][0]["geometry"]["coordinates"]
    route_latlon = [(lat, lon) for lon, lat in route]  # flip order
    m = folium.Map(location=route_latlon[0], zoom_start=12)
    folium.Marker(route_latlon[0], tooltip="Start: Gandhipuram", icon=folium.Icon(color="green")).add_to(m)
    folium.Marker(route_latlon[-1], tooltip="End: Thirumalayampalayam", icon=folium.Icon(color="red")).add_to(m)
    folium.PolyLine(route_latlon, weight=5).add_to(m)
    return m

# ========== SHOW MAP SAFELY ==========
def show_map(m):
    """Ensures Folium map works in Jupyter, VSCode, or JupyterLab"""
    try:
        display(m)  # Works in Jupyter Notebook
    except:
        m.save("route_map.html")
        display(IFrame("route_map.html", width=800, height=600))

# ========== RUN ==========
resp = osrm_route(OSRM_URL, src, dst, PROFILE)
if resp:
    print(pretty_route_summary(resp))
    m = route_to_folium(resp)
    show_map(m)
