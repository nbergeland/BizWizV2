# === PART 3: QUICK START SCRIPT ===
# Save this as: run_analysis.py

"""
Quick start script to run the entire analysis.
This will:
1. Check if data has been collected
2. Run data collection if needed
3. Start the visualization app
"""

import os
import subprocess
import sys

def main():
    processed_data_file = 'processed_location_data.pkl'
    
    print("🐔 Raising Cane's Location Analysis Tool")
    print("=" * 50)
    
    # Check if processed data exists
    if not os.path.exists(processed_data_file):
        print("📊 No processed data found. Starting data collection...")
        print("⚠️  This will make API calls and may take several minutes.")
        
        response = input("Continue with data collection? (y/n): ").lower().strip()
        if response != 'y':
            print("Exiting...")
            return
        
        # Run data collection
        print("🔄 Running data collection...")
        try:
            from data_collection import collect_and_process_all_data
            collect_and_process_all_data()
            print("✅ Data collection completed!")
        except ImportError:
            print("❌ Error: Could not import data_collection module.")
            print("Make sure data_collection.py is in the same directory.")
            return
        except Exception as e:
            print(f"❌ Error during data collection: {e}")
            return
    else:
        print("✅ Found existing processed data.")
    
    # Start visualization app
    print("🚀 Starting visualization app...")
    try:
        from visualization_app import app
        print("🌐 Open your browser to: http://127.0.0.1:8050")
        app.run(debug=False, host='127.0.0.1', port=8050)
    except ImportError:
        print("❌ Error: Could not import visualization_app module.")
        print("Make sure visualization_app.py is in the same directory.")
    except Exception as e:
        print(f"❌ Error starting app: {e}")

if __name__ == '__main__':
    main()


# === INSTALLATION AND USAGE INSTRUCTIONS ===

"""
SETUP INSTRUCTIONS:
==================

1. Install required packages:
   pip install pandas numpy googlemaps requests scikit-learn plotly dash dash-bootstrap-components python-dotenv

2. Create a .env file with your API keys:
   GOOGLE_API_KEY=your_google_maps_api_key
   CENSUS_API_KEY=your_census_api_key  
   RENTCAST_API_KEY=your_rentcast_api_key

3. Save the code as three separate files:
   - data_collection.py (Part 1)
   - visualization_app.py (Part 2) 
   - run_analysis.py (Part 3)

4. Run the analysis:
   python run_analysis.py

WHAT'S NEW:
===========

✅ Separated API calls from visualization
✅ Added Chick-fil-A and Raising Cane's location fetching
✅ Added chicken emoji markers (🐔) for Chick-fil-A locations
✅ Added chicken leg emoji markers (🍗) for existing Raising Cane's
✅ Data is cached and only collected once
✅ Easy-to-use quick start script

MAP FEATURES:
=============

🔴 Red to Green dots = Potential new Raising Cane's locations (revenue potential)
🐔 Red chicken emojis = Chick-fil-A locations (competition reference)
🍗 Purple chicken leg emojis = Existing Raising Cane's locations
📊 Interactive filters for analysis
📈 Revenue predictions based on commercial factors

Perfect for identifying optimal locations for new Raising Cane's restaurants
while seeing where the competition (Chick-fil-A) is located!
"""

