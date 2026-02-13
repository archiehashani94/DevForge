# Industrial Land Monitoring System

A professional Streamlit-based GIS dashboard for monitoring industrial plot boundaries and detecting violations.

## Features

### 🗺️ Interactive Map
- **CSIDC Portal Integration**: Reference marker linking to official CGGIS portal
- **Dual Base Layers**: Satellite imagery and street map views
- **Plot Overlays**: Color-coded polygons based on compliance status
- **Interactive Popups**: Click plots to view detailed information
- **Fullscreen Mode**: Expand map for better viewing

### 📊 Dashboard Analytics
- **Statistics Cards**: Total plots, compliant, encroachments, vacant
- **Compliance Rate**: Real-time compliance percentage
- **Visual Indicators**: Color-coded metrics

### ⚠️ Violation Detection
- **Status-Based Coloring**:
  - 🟢 Green = Compliant
  - 🔴 Red = Encroachment
  - 🟡 Yellow = Vacant/Unused
- **Alerts Table**: Detailed list of problematic plots
- **Severity Levels**: High, Medium classification
- **CSV Export**: Download alerts report

### 🎛️ Controls
- **Status Filter**: Filter plots by compliance status
- **File Upload**: Upload new GeoJSON data
- **Layer Control**: Toggle between base maps

## Installation

1. **Navigate to project directory:**
```bash
cd industrial_land_monitoring
```

2. **Create virtual environment:**
```bash
python -m venv venv
venv\Scripts\activate  # Windows
```

3. **Install dependencies:**
```bash
pip install -r requirements.txt
```

## Usage

**Start the application:**
```bash
streamlit run app.py
```

The dashboard will open in your browser at `http://localhost:8501`

## Data Files

### `data/plots.geojson`
Contains industrial plot boundaries with properties:
- `plot_id`: Unique identifier
- `owner`: Plot owner name
- `area`: Area in square meters
- `zone`: Industrial zone designation
- `status`: Current status
- `last_updated`: Last inspection date

### `data/results.geojson`
Contains inspection results with:
- `plot_id`: Matching plot identifier
- `status`: compliant | encroachment | vacant
- `compliance_score`: 0-100 score
- `violation_type`: Type of violation (if any)
- `severity`: high | medium | low

## Project Structure

```
industrial_land_monitoring/
├── app.py                 # Main Streamlit application
├── requirements.txt       # Python dependencies
├── data/
│   ├── plots.geojson     # Plot boundaries
│   └── results.geojson   # Inspection results
└── README.md
```

## Features in Detail

### Plot Information Popup
Each plot shows:
- Plot ID
- Owner name
- Area (sq.m)
- Industrial zone
- Compliance status
- Compliance score
- Last update date
- Issue description (if applicable)

### Alerts Table
Displays:
- Plot ID
- Issue type
- Severity level
- Current status

### File Upload
- Upload new `results.geojson` files
- Instantly updates map and statistics
- Supports standard GeoJSON format

## Technology Stack

- **Frontend**: Streamlit
- **Mapping**: Folium with Leaflet.js
- **Data Processing**: GeoPandas, Pandas
- **Base Maps**: ESRI Satellite, OpenStreetMap

## CSIDC Integration

The dashboard includes a reference marker linking to the official CGGIS CSIDC portal:
https://cggis.cgstate.gov.in/csidc/

## Sample Data

The project includes sample data for 8 industrial plots around Nava Raipur, Chhattisgarh:
- 4 Compliant plots
- 2 Encroachment cases
- 2 Vacant plots

## Customization

### Adding New Plots
1. Edit `data/plots.geojson`
2. Add GeoJSON features with required properties
3. Ensure coordinates are in WGS84 (EPSG:4326)

### Updating Results
1. Edit `data/results.geojson`
2. Match `plot_id` with plots.geojson
3. Set appropriate status and scores

### Changing Map Center
Edit `app.py` line ~200:
```python
center = [21.16, 81.78]  # [latitude, longitude]
```

## License

Government Project - CSIDC

## Contact

For questions or support, contact the CSIDC development team.
