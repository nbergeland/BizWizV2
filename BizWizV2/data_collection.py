# === PART 1: DATA COLLECTION SCRIPT (run once) ===
# Save this as: data_collection.py

import os
import numpy as np
import pandas as pd
import googlemaps
import requests
import time
import json
import datetime
from sklearn.ensemble import RandomForestRegressor
from math import radians, cos, sin, asin, sqrt
from dotenv import load_dotenv
import pickle
from functools import lru_cache

# === LOAD .env VARIABLES ===
load_dotenv()
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')
CENSUS_API_KEY = os.getenv('CENSUS_API_KEY')
RENTCAST_API_KEY = os.getenv('RENTCAST_API_KEY')

# === GOOGLE MAPS CLIENT ===
gmaps = googlemaps.Client(key=GOOGLE_API_KEY)

# === GRID GENERATION - GRAND FORKS, ND ===
min_lat, max_lat = 47.85, 47.95
min_lon, max_lon = -97.15, -97.0
grid_spacing = 0.005
lats = np.arange(min_lat, max_lat, grid_spacing)
lons = np.arange(min_lon, max_lon, grid_spacing)
grid_points = [(lat, lon) for lat in lats for lon in lons]

# === CACHING SETUP ===
CACHE_FILE = 'location_data_cache.pkl'
USAGE_FILE = 'api_usage.json'
PROCESSED_DATA_FILE = 'processed_location_data.pkl'

def load_cache():
    try:
        with open(CACHE_FILE, 'rb') as f:
            return pickle.load(f)
    except FileNotFoundError:
        return {}

def save_cache(cache):
    with open(CACHE_FILE, 'wb') as f:
        pickle.dump(cache, f)

def load_api_usage():
    try:
        with open(USAGE_FILE, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        return {'daily_calls': 0, 'date': str(datetime.date.today())}

def save_api_usage(usage):
    with open(USAGE_FILE, 'w') as f:
        json.dump(usage, f)

def track_api_call():
    """Track API calls to monitor usage"""
    usage = load_api_usage()
    today = str(datetime.date.today())
    
    if usage['date'] != today:
        usage = {'daily_calls': 0, 'date': today}
    
    usage['daily_calls'] += 1
    save_api_usage(usage)
    
    print(f"API calls today: {usage['daily_calls']}")
    return usage['daily_calls']

# === DISTANCE FUNCTION IN MILES ===
def calculate_distance_miles(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    a = sin((lat2-lat1)/2)**2 + cos(lat1) * cos(lat2) * sin((lon2-lon1)/2)**2
    c = 2 * asin(sqrt(a))
    return c * 3956

# === ROAD DATA FROM OPENSTREETMAP (FREE) ===
def fetch_road_data():
    """Fetch major roads using OpenStreetMap Overpass API (free)"""
    cache_key = 'osm_roads'
    cache = load_cache()
    
    if cache_key in cache:
        return cache[cache_key]
    
    overpass_url = "http://overpass-api.de/api/interpreter"
    overpass_query = f"""
    [out:json][timeout:25];
    (
      way["highway"~"^(trunk|primary|secondary|trunk_link|primary_link)$"]({min_lat},{min_lon},{max_lat},{max_lon});
    );
    out geom;
    """
    
    try:
        print("Fetching road data from OpenStreetMap...")
        response = requests.get(overpass_url, params={'data': overpass_query})
        roads_data = response.json()
        
        # Extract road coordinates
        road_points = []
        for way in roads_data.get('elements', []):
            if 'geometry' in way:
                for point in way['geometry']:
                    road_points.append((point['lat'], point['lon']))
        
        cache[cache_key] = road_points
        save_cache(cache)
        print(f"Found {len(road_points)} road points")
        return road_points
        
    except Exception as e:
        print(f"Error fetching road data: {e}")
        return []

# === IMPROVED DATA FETCHER ===
class CommercialLocationDataFetcher:
    def __init__(self):
        self.cache = load_cache()
        self.chickfila_locations = None
        self.raising_canes_locations = None  # NEW: Add Raising Cane's locations
        self.competitor_locations = {}
        self.poi_locations = {}
        self.active_listings = []
        self.road_points = []
        
    def fetch_all_chickfila_locations(self):
        """Fetch all Chick-fil-A locations in the broader area once"""
        if self.chickfila_locations is not None:
            return
            
        cache_key = 'chickfila_all'
        if cache_key in self.cache:
            self.chickfila_locations = self.cache[cache_key]
            return
            
        center_lat = (min_lat + max_lat) / 2
        center_lon = (min_lon + max_lon) / 2
        
        try:
            track_api_call()
            result = gmaps.places_nearby(
                location=(center_lat, center_lon), 
                radius=50000,  # 50km radius
                keyword='chick-fil-a'
            )
            locations = result['results']
            
            # Handle pagination if needed
            while 'next_page_token' in result:
                time.sleep(2)
                track_api_call()
                result = gmaps.places_nearby(
                    location=(center_lat, center_lon),
                    radius=50000,
                    keyword='chick-fil-a',
                    page_token=result['next_page_token']
                )
                locations.extend(result['results'])
                
            self.chickfila_locations = [(
                loc['geometry']['location']['lat'],
                loc['geometry']['location']['lng']
            ) for loc in locations]
            
            self.cache[cache_key] = self.chickfila_locations
            save_cache(self.cache)
            
        except Exception as e:
            print(f"Error fetching Chick-fil-A locations: {e}")
            self.chickfila_locations = []

    def fetch_all_raising_canes_locations(self):
        """Fetch all Raising Cane's locations in the broader area once"""
        if self.raising_canes_locations is not None:
            return
            
        cache_key = 'raising_canes_all'
        if cache_key in self.cache:
            self.raising_canes_locations = self.cache[cache_key]
            return
            
        center_lat = (min_lat + max_lat) / 2
        center_lon = (min_lon + max_lon) / 2
        
        try:
            track_api_call()
            result = gmaps.places_nearby(
                location=(center_lat, center_lon), 
                radius=50000,  # 50km radius
                keyword="raising cane's"
            )
            locations = result['results']
            
            # Handle pagination if needed
            while 'next_page_token' in result:
                time.sleep(2)
                track_api_call()
                result = gmaps.places_nearby(
                    location=(center_lat, center_lon),
                    radius=50000,
                    keyword="raising cane's",
                    page_token=result['next_page_token']
                )
                locations.extend(result['results'])
                
            self.raising_canes_locations = [(
                loc['geometry']['location']['lat'],
                loc['geometry']['location']['lng'],
                loc.get('name', "Raising Cane's")
            ) for loc in locations]
            
            self.cache[cache_key] = self.raising_canes_locations
            save_cache(self.cache)
            
        except Exception as e:
            print(f"Error fetching Raising Cane's locations: {e}")
            self.raising_canes_locations = []
    
    def fetch_competitor_locations(self):
        """Fetch all competitor locations once"""
        if self.competitor_locations:
            return
            
        center_lat = (min_lat + max_lat) / 2
        center_lon = (min_lon + max_lon) / 2
        competitors = ['mcdonalds', 'kfc', 'taco bell', 'burger king', 'subway', 'wendys', 'popeyes']
        
        for competitor in competitors:
            cache_key = f'competitor_{competitor}'
            if cache_key in self.cache:
                self.competitor_locations[competitor] = self.cache[cache_key]
                continue
                
            try:
                track_api_call()
                result = gmaps.places_nearby(
                    location=(center_lat, center_lon),
                    radius=20000,  # 20km radius
                    keyword=competitor
                )
                locations = [(
                    loc['geometry']['location']['lat'],
                    loc['geometry']['location']['lng']
                ) for loc in result['results']]
                
                self.competitor_locations[competitor] = locations
                self.cache[cache_key] = locations
                time.sleep(0.2)  # Rate limiting
                
            except Exception as e:
                print(f"Error fetching {competitor} locations: {e}")
                self.competitor_locations[competitor] = []
        
        save_cache(self.cache)
    
    def fetch_commercial_poi_locations(self):
        """Fetch commercial-focused points of interest"""
        if self.poi_locations:
            return
            
        center_lat = (min_lat + max_lat) / 2
        center_lon = (min_lon + max_lon) / 2
        
        # Commercial-focused POI with higher weights for business viability
        poi_types = [
            ('shopping_mall', 'shopping_mall', 50),      # High traffic generators
            ('gas_station', 'gas_station', 30),         # High visibility locations
            ('bank', 'bank', 25),                       # Commercial corridors
            ('pharmacy', 'pharmacy', 20),               # Strip mall locations
            ('supermarket', 'supermarket', 40),         # Anchor stores
            ('hospital', 'hospital', 30),               # High traffic
            ('university', 'university', 35),           # Student traffic
            ('gym', 'gym', 15),                         # Commercial areas
            ('car_dealer', 'car_dealer', 20),           # Commercial strips
            ('lodging', 'lodging', 25),                 # Commercial zones
            ('store', 'store', 10),                     # General retail
            ('restaurant', 'restaurant', 5)             # Food service areas
        ]
        
        for poi_name, poi_type, weight in poi_types:
            cache_key = f'poi_{poi_name}'
            if cache_key in self.cache:
                self.poi_locations[poi_name] = self.cache[cache_key]
                continue
                
            try:
                track_api_call()
                if poi_name == 'university':
                    result = gmaps.places_nearby(
                        location=(center_lat, center_lon),
                        radius=25000,  # Larger radius for better coverage
                        keyword='university'
                    )
                else:
                    result = gmaps.places_nearby(
                        location=(center_lat, center_lon),
                        radius=25000,
                        type=poi_type
                    )
                    
                locations = [(
                    loc['geometry']['location']['lat'],
                    loc['geometry']['location']['lng'],
                    weight
                ) for loc in result['results']]
                
                self.poi_locations[poi_name] = locations
                self.cache[cache_key] = locations
                time.sleep(0.2)
                
            except Exception as e:
                print(f"Error fetching {poi_name} locations: {e}")
                self.poi_locations[poi_name] = []
        
        save_cache(self.cache)

    def fetch_rental_listings(self):
        """Fetch rental listings once"""
        if self.active_listings:
            return
            
        cache_key = 'rental_listings'
        if cache_key in self.cache:
            # Check if cache is recent (less than 24 hours old)
            cache_time = self.cache.get(f'{cache_key}_timestamp', 0)
            if time.time() - cache_time < 86400:  # 24 hours
                self.active_listings = self.cache[cache_key]
                return
        
        try:
            url = "https://api.rentcast.io/v1/listings/rental/long-term"
            headers = {"X-Api-Key": RENTCAST_API_KEY}
            params = {"city": "Grand Forks", "state": "ND", "status": "active", "limit": 500}
            response = requests.get(url, headers=headers, params=params)
            
            if response.status_code == 200:
                self.active_listings = response.json().get('listings', [])
                self.cache[cache_key] = self.active_listings
                self.cache[f'{cache_key}_timestamp'] = time.time()
                save_cache(self.cache)
            else:
                self.active_listings = []
                
        except Exception as e:
            print(f"Error fetching rental listings: {e}")
            self.active_listings = []

    def fetch_road_data(self):
        """Load road data from OpenStreetMap"""
        self.road_points = fetch_road_data()

    @lru_cache(maxsize=1000)
    def get_demographics_cached(self, lat_rounded, lon_rounded):
        """Cache demographics by rounded coordinates to avoid duplicate census calls"""
        try:
            fcc_url = f"https://geo.fcc.gov/api/census/block/find?latitude={lat_rounded}&longitude={lon_rounded}&format=json"
            fcc_response = requests.get(fcc_url)
            fips = fcc_response.json()['Block']['FIPS'][:11]
            
            url = f"https://api.census.gov/data/2020/acs/acs5?get=B01003_001E,B19013_001E,B01002_001E&for=tract:{fips[5:11]}&in=state:{fips[:2]}+county:{fips[2:5]}&key={CENSUS_API_KEY}"
            data = requests.get(url).json()[1]
            
            return {
                'population': int(data[0]), 
                'median_income': int(data[1]), 
                'median_age': float(data[2])
            }
        except:
            return {'population': 5000, 'median_income': 45000, 'median_age': 28}

    def calculate_commercial_viability_score(self, lat, lon):
        """Calculate commercial viability without additional API calls"""
        
        # Commercial foot traffic score (much higher weights)
        commercial_traffic = 0
        for poi_type, locations in self.poi_locations.items():
            if poi_type in ['shopping_mall', 'supermarket', 'gas_station', 'bank']:
                for p_lat, p_lon, weight in locations:
                    distance = calculate_distance_miles(lat, lon, p_lat, p_lon)
                    if distance <= 1:  # Within 1 mile
                        commercial_traffic += weight
        
        # Visibility/accessibility score using pre-fetched road data
        road_accessibility = 0
        if self.road_points:
            nearby_roads = [
                1 for r_lat, r_lon in self.road_points
                if calculate_distance_miles(lat, lon, r_lat, r_lon) <= 0.2  # Within 0.2 miles
            ]
            road_accessibility = min(len(nearby_roads) * 5, 50)  # Cap at 50
        
        # Gas station proximity (indicator of major roads and visibility)
        gas_stations = self.poi_locations.get('gas_station', [])
        gas_proximity = 0
        for g_lat, g_lon, _ in gas_stations:
            distance = calculate_distance_miles(lat, lon, g_lat, g_lon)
            if distance <= 0.5:  # Within 0.5 miles
                gas_proximity += 15
        
        return {
            'commercial_traffic_score': commercial_traffic,
            'road_accessibility_score': road_accessibility,
            'gas_station_proximity': gas_proximity
        }

    def detect_residential_bias(self, lat, lon, active_listings_count, population):
        """Detect if location is heavily residential"""
        
        residential_indicators = 0
        
        # High apartment density
        if active_listings_count > 15:
            residential_indicators += 10
        
        # Very high population density (typical of residential areas)
        if population > 8000:
            residential_indicators += 15
        
        # Low commercial activity
        commercial_nearby = 0
        for poi_type in ['gas_station', 'bank', 'shopping_mall']:
            locations = self.poi_locations.get(poi_type, [])
            for p_lat, p_lon, _ in locations:
                if calculate_distance_miles(lat, lon, p_lat, p_lon) <= 0.5:
                    commercial_nearby += 1
        
        if commercial_nearby == 0:
            residential_indicators += 10
        
        return residential_indicators

    def calculate_features_for_point(self, lat, lon):
        """Calculate all features for a single point using cached data"""
        # Round coordinates for demographic caching
        lat_rounded = round(lat, 3)
        lon_rounded = round(lon, 3)
        
        # Chick-fil-A proximity
        if self.chickfila_locations:
            distances_to_chickfila = [
                calculate_distance_miles(lat, lon, c_lat, c_lon) 
                for c_lat, c_lon in self.chickfila_locations
            ]
            chick_count = len([d for d in distances_to_chickfila if d <= 5])
            nearest_chickfila = min(distances_to_chickfila) if distances_to_chickfila else 30
        else:
            chick_count, nearest_chickfila = 0, 30
        
        # Fast food competition
        competition_count = 0
        for competitor, locations in self.competitor_locations.items():
            nearby_competitors = [
                1 for c_lat, c_lon in locations 
                if calculate_distance_miles(lat, lon, c_lat, c_lon) <= 2
            ]
            competition_count += len(nearby_competitors)
        
        # Commercial viability scores
        commercial_scores = self.calculate_commercial_viability_score(lat, lon)
        
        # Demographics (cached)
        demographics = self.get_demographics_cached(lat_rounded, lon_rounded)
        
        # Rental data
        nearby_listings = []
        for listing in self.active_listings:
            if listing.get('latitude') and listing.get('longitude'):
                distance = calculate_distance_miles(
                    lat, lon, listing['latitude'], listing['longitude']
                )
                if distance <= 1:
                    nearby_listings.append(listing['price'])
        
        active_listings_count = len(nearby_listings)
        avg_rent = np.mean(nearby_listings) if nearby_listings else 0
        
        # Residential bias detection
        residential_bias = self.detect_residential_bias(
            lat, lon, active_listings_count, demographics['population']
        )
        
        # Zoning (randomized for demo)
        import random
        zoning = random.choice([True, False])
        
        return {
            'latitude': lat,
            'longitude': lon,
            'chickfila_count_nearby': chick_count,
            'distance_to_chickfila': nearest_chickfila,
            'fast_food_competition': competition_count,
            'commercial_traffic_score': commercial_scores['commercial_traffic_score'],
            'road_accessibility_score': commercial_scores['road_accessibility_score'],
            'gas_station_proximity': commercial_scores['gas_station_proximity'],
            'population': demographics['population'],
            'median_income': demographics['median_income'],
            'median_age': demographics['median_age'],
            'rent_per_sqft': 12.50,
            'zoning_compliant': int(zoning),
            'active_listings_within_1_mile': active_listings_count,
            'average_nearby_rent': avg_rent,
            'residential_bias_score': residential_bias
        }

# === MAIN DATA COLLECTION FUNCTION ===
def collect_and_process_all_data():
    """Main function to collect all data and save processed results"""
    
    # Check if processed data already exists
    if os.path.exists(PROCESSED_DATA_FILE):
        print("Processed data file already exists. Loading existing data...")
        with open(PROCESSED_DATA_FILE, 'rb') as f:
            data = pickle.load(f)
        return data['df_filtered'], data['model'], data['feature_importance'], data['chickfila_locations'], data['raising_canes_locations']
    
    print("Starting commercial location analysis...")
    fetcher = CommercialLocationDataFetcher()
    
    print("Fetching bulk data...")
    fetcher.fetch_all_chickfila_locations()
    print(f"Found {len(fetcher.chickfila_locations)} Chick-fil-A locations")
    
    # NEW: Fetch Raising Cane's locations (for reference/competitive analysis)
    fetcher.fetch_all_raising_canes_locations()
    print(f"Found {len(fetcher.raising_canes_locations)} Raising Cane's locations")
    
    fetcher.fetch_competitor_locations()
    total_competitors = sum(len(locs) for locs in fetcher.competitor_locations.values())
    print(f"Found {total_competitors} competitor locations")
    
    fetcher.fetch_commercial_poi_locations()
    total_pois = sum(len(locs) for locs in fetcher.poi_locations.values())
    print(f"Found {total_pois} points of interest")
    
    fetcher.fetch_rental_listings()
    print(f"Found {len(fetcher.active_listings)} rental listings")
    
    fetcher.fetch_road_data()
    print(f"Found {len(fetcher.road_points)} road points")
    
    print("Processing grid points...")
    feature_list = []
    
    for idx, (lat, lon) in enumerate(grid_points):
        if idx % 10 == 0:
            print(f"Processing {idx+1}/{len(grid_points)}: {lat:.4f}, {lon:.4f}")
        
        features = fetcher.calculate_features_for_point(lat, lon)
        feature_list.append(features)
        
        # Minimal delay
        if idx % 100 == 0:
            time.sleep(0.1)
    
    df = pd.DataFrame(feature_list)
    
    # Calculate derived commercial features
    df['chick_fil_a_advantage'] = np.where(
        (df['distance_to_chickfila'] > 2) & (df['distance_to_chickfila'] < 8), 
        800 / df['distance_to_chickfila'], 
        0
    )

    # Youth factor (important for fast-casual dining)
    df['youth_factor'] = np.where(df['median_age'] < 35, 800, 0)

    # Competition clustering advantage (some competition is good)
    df['competitive_cluster_bonus'] = np.where(
        (df['fast_food_competition'] >= 2) & (df['fast_food_competition'] <= 6), 
        300, 
        np.where(df['fast_food_competition'] > 6, -200, 0)
    )

    # Commercial-focused revenue calculation
    df['estimated_revenue'] = (
        # Commercial factors (high weights)
        df['commercial_traffic_score'] * 150 +
        df['road_accessibility_score'] * 100 +
        df['gas_station_proximity'] * 80 +
        df['competitive_cluster_bonus'] +
        
        # Demographics (moderate weights, focus on income)
        df['median_income'] * 0.002 +
        df['youth_factor'] +
        
        # Strategic positioning
        df['chick_fil_a_advantage'] * 400 +
        
        # Penalties for residential bias
        df['active_listings_within_1_mile'] * -100 +  # Penalty for apartment density
        df['residential_bias_score'] * -150 +         # General residential penalty
        np.where(df['population'] > 9000, -500, 0) +  # Very dense residential penalty
        
        # Zoning compliance bonus
        df['zoning_compliant'] * 1200 +
        
        # Base commercial viability
        2000
    )

    # Ensure non-negative revenue
    df['estimated_revenue'] = np.maximum(df['estimated_revenue'], 0)

    # Filter out locations with high residential bias unless they have strong commercial indicators
    df['keep_location'] = (
        (df['residential_bias_score'] < 20) |  # Low residential bias, or
        (df['commercial_traffic_score'] > 50)  # Strong commercial indicators
    )

    # Apply filter
    df_filtered = df[df['keep_location']].copy()

    print(f"Filtered out {len(df) - len(df_filtered)} residential-biased locations")
    print(f"Kept {len(df_filtered)} commercially viable locations")

    # Model training on filtered data
    X = df_filtered.drop(columns=[
        'latitude', 'longitude', 'estimated_revenue', 'keep_location'
    ]).select_dtypes(include=[np.number])
    X = X.replace([np.inf, -np.inf], np.nan).fillna(X.mean())
    y = df_filtered['estimated_revenue']

    model = RandomForestRegressor(n_estimators=200, random_state=42, max_depth=12)
    model.fit(X, y)
    df_filtered['predicted_revenue'] = model.predict(X)

    # Feature importance analysis
    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)

    print("\nTop 10 Most Important Features:")
    print(feature_importance.head(10))

    print(f"\nCommercial location analysis complete. Processed {len(df_filtered)} locations.")
    print(f"Top predicted revenue: ${df_filtered['predicted_revenue'].max():,.0f}")
    
    # Save all processed data
    processed_data = {
        'df_filtered': df_filtered,
        'model': model,
        'feature_importance': feature_importance,
        'chickfila_locations': fetcher.chickfila_locations,
        'raising_canes_locations': fetcher.raising_canes_locations
    }
    
    with open(PROCESSED_DATA_FILE, 'wb') as f:
        pickle.dump(processed_data, f)
    
    print(f"Processed data saved to {PROCESSED_DATA_FILE}")
    
    return data['df_filtered'], data['model'], data['feature_importance'], data['chickfila_locations'], data['raising_canes_locations']

# Run data collection if this script is executed directly
if __name__ == '__main__':
    collect_and_process_all_data()
