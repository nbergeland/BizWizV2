# === PART 2: VISUALIZATION SCRIPT (run after data collection) ===
# Save this as: visualization_app.py

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from dash import Dash, dcc, html, Input, Output, State
import dash_bootstrap_components as dbc
import pickle
import os

# Load processed data
PROCESSED_DATA_FILE = 'processed_location_data.pkl'

def load_processed_data():
    """Load the processed data from the data collection script"""
    if not os.path.exists(PROCESSED_DATA_FILE):
        print(f"Error: {PROCESSED_DATA_FILE} not found!")
        print("Please run the data collection script first (data_collection.py)")
        return None, None, None, None, None
    
    with open(PROCESSED_DATA_FILE, 'rb') as f:
        data = pickle.load(f)
    
    return data['df_filtered'], data['model'], data['feature_importance'], data['chickfila_locations'], data['raising_canes_locations']

# Load data
df_filtered, model, feature_importance, chickfila_locations, raising_canes_locations = load_processed_data()

if df_filtered is None:
    print("Cannot start app without processed data. Exiting...")
    exit(1)

print(f"Loaded {len(df_filtered)} processed locations")
print(f"Loaded {len(chickfila_locations)} Chick-fil-A locations") 
print(f"Loaded {len(raising_canes_locations)} Raising Cane's locations")

# === DASH APP ===
app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

app.layout = dbc.Container([
    dbc.Row(dbc.Col(html.H2("Optimal Raising Cane's Locations in Grand Forks, ND"), className="text-center my-4")),
    
    dbc.Row([
        dbc.Col([
            html.Label("Minimum Predicted Revenue:"),
            dcc.Slider(
                id='revenue-slider', 
                min=0, 
                max=df_filtered['predicted_revenue'].max(), 
                step=1000, 
                value=df_filtered['predicted_revenue'].quantile(0.6),
                tooltip={"placement": "bottom", "always_visible": True}
            ),
            
            html.Label("Maximum Distance to Chick-fil-A (miles):"),
            dcc.Slider(
                id='chickfila-distance-slider', 
                min=0, 
                max=15, 
                step=1, 
                value=8,
                tooltip={"placement": "bottom", "always_visible": True}
            ),
            
            html.Label("Minimum Commercial Traffic Score:"),
            dcc.Slider(
                id='commercial-traffic-slider', 
                min=0, 
                max=df_filtered['commercial_traffic_score'].max(), 
                step=10, 
                value=20,
                tooltip={"placement": "bottom", "always_visible": True}
            ),
            
            html.Label("Maximum Fast Food Competition:"),
            dcc.Slider(
                id='competition-slider', 
                min=0, 
                max=15, 
                step=1, 
                value=8,
                tooltip={"placement": "bottom", "always_visible": True}
            ),
            
            html.Label("Zoning Compliance:"),
            dcc.RadioItems(
                id='zoning-radio', 
                options=[
                    {'label': 'All Locations', 'value': 'all'}, 
                    {'label': 'Only Compliant', 'value': 'compliant'}
                ], 
                value='compliant'
            ),
            
            html.Div(id='location-stats', className="mt-4 p-3 bg-light rounded")
        ], width=3),
        
        dbc.Col([
            dcc.Graph(id='revenue-map', style={'height': '80vh'})
        ], width=9)
    ])
], fluid=True)

@app.callback(
    [Output('revenue-map', 'figure'), Output('location-stats', 'children')],
    [Input('revenue-slider', 'value'), 
     Input('chickfila-distance-slider', 'value'), 
     Input('commercial-traffic-slider', 'value'),
     Input('competition-slider', 'value'),
     Input('zoning-radio', 'value')]
)
def update_map(min_revenue, max_chickfila_distance, min_commercial_traffic, max_competition, zoning_filter):
    filtered = df_filtered[
        (df_filtered['predicted_revenue'] >= min_revenue) &
        (df_filtered['distance_to_chickfila'] <= max_chickfila_distance) &
        (df_filtered['commercial_traffic_score'] >= min_commercial_traffic) &
        (df_filtered['fast_food_competition'] <= max_competition)
    ]
    
    if zoning_filter == 'compliant':
        filtered = filtered[filtered['zoning_compliant'] == 1]
    
    # Create base scatter plot for potential locations
    fig = px.scatter_mapbox(
        filtered, 
        lat='latitude', 
        lon='longitude', 
        size='predicted_revenue', 
        color='predicted_revenue',
        color_continuous_scale='RdYlGn', 
        size_max=25, 
        zoom=12, 
        mapbox_style='carto-positron',
        hover_data=['commercial_traffic_score', 'road_accessibility_score', 'distance_to_chickfila'],
        title="Commercial Locations Ranked by Revenue Potential"
    )
    
    # Add Chick-fil-A locations with chicken emojis
    if chickfila_locations and len(chickfila_locations) > 0:
        # Create DataFrame for Chick-fil-A locations
        chickfila_df = pd.DataFrame(chickfila_locations, columns=['latitude', 'longitude'])
        
        # Add Chick-fil-A locations as a separate trace with chicken emoji
        fig.add_trace(
            go.Scattermapbox(
                lat=chickfila_df['latitude'],
                lon=chickfila_df['longitude'],
                mode='markers+text',
                marker=dict(
                    size=20,
                    color='red',
                    symbol='circle'
                ),
                text='🐔',  # Chicken emoji for Chick-fil-A
                textfont=dict(size=16),
                textposition='middle center',
                name="Chick-fil-A Locations",
                hovertemplate="<b>Chick-fil-A</b><br>" +
                             "Lat: %{lat:.4f}<br>" +
                             "Lon: %{lon:.4f}<br>" +
                             "<extra></extra>"
            )
        )
    
    # Add existing Raising Cane's locations (if any) with different marker
    if raising_canes_locations and len(raising_canes_locations) > 0:
        # Create DataFrame for existing Raising Cane's locations
        canes_df = pd.DataFrame(raising_canes_locations, columns=['latitude', 'longitude', 'name'])
        
        # Add Raising Cane's locations as a separate trace
        fig.add_trace(
            go.Scattermapbox(
                lat=canes_df['latitude'],
                lon=canes_df['longitude'],
                mode='markers+text',
                marker=dict(
                    size=20,
                    color='purple',
                    symbol='circle'
                ),
                text='🍗',  # Chicken leg emoji for Raising Cane's
                textfont=dict(size=16),
                textposition='middle center',
                name="Existing Raising Cane's",
                hovertemplate="<b>Existing Raising Cane's</b><br>" +
                             "Location: %{customdata}<br>" +
                             "<extra></extra>",
                customdata=canes_df['name']
            )
        )
    
    # Update layout
    fig.update_layout(
        height=600,
        showlegend=True,
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01,
            bgcolor="rgba(255,255,255,0.8)"
        )
    )
    
    # Location statistics
    if len(filtered) > 0:
        best = filtered.loc[filtered['predicted_revenue'].idxmax()]
        avg_revenue = filtered['predicted_revenue'].mean()
        
        stats = html.Div([
            html.H5("Analysis Summary", className="text-primary"),
            html.P(f"Potential Locations: {len(filtered)}"),
            html.P(f"Chick-fil-A Locations: {len(chickfila_locations)}"),
            html.P(f"Existing Raising Cane's: {len(raising_canes_locations)}"),
            html.P(f"Average Revenue: ${avg_revenue:,.0f}"),
            html.Hr(),
            html.H5("Top Potential Location", className="text-success"),
            html.P(f"📍 {best['latitude']:.4f}, {best['longitude']:.4f}"),
            html.P(f"💰 Revenue: ${best['predicted_revenue']:,.0f}"),
            html.P(f"🏪 Commercial Score: {best['commercial_traffic_score']:.0f}"),
            html.P(f"🛣️ Road Access: {best['road_accessibility_score']:.0f}"),
            html.P(f"⛽ Gas Station Proximity: {best['gas_station_proximity']:.0f}"),
            html.P(f"🍗 Distance to Chick-fil-A: {best['distance_to_chickfila']:.1f} mi"),
            html.P(f"🏢 Competition: {best['fast_food_competition']:.0f}"),
            html.P(f"👥 Median Age: {best['median_age']:.0f}"),
            html.P(f"💵 Median Income: ${best['median_income']:,.0f}")
        ])
    else:
        stats = html.Div([
            html.H5("No Locations Found", className="text-warning"),
            html.P("Try adjusting your filters to see more locations."),
            html.P(f"Chick-fil-A Locations: {len(chickfila_locations)}"),
            html.P(f"Existing Raising Cane's: {len(raising_canes_locations)}")
        ])
    
    return fig, stats

if __name__ == '__main__':
    app.run(debug=True)


