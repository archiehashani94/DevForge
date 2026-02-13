"""
Industrial Land Monitoring System
A Streamlit-based GIS dashboard for monitoring industrial plot boundaries
and detecting violations like encroachments and unused plots.
"""

import streamlit as st
import folium
from folium import plugins
from streamlit_folium import st_folium
import json
import pandas as pd
from pathlib import Path
import geopandas as gpd
from datetime import datetime

# Page configuration
st.set_page_config(
    page_title="Industrial Land Monitoring System",
    page_icon="🏭",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for professional government dashboard
st.markdown("""
<style>
    /* Main theme */
    .main {
        background-color: #f5f7fa;
    }
    
    /* Header styling */
    .dashboard-header {
        background: linear-gradient(135deg, #1e3a8a 0%, #1e40af 100%);
        color: white;
        padding: 1.5rem 2rem;
        border-radius: 10px;
        margin-bottom: 2rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    
    .dashboard-title {
        font-size: 2rem;
        font-weight: 700;
        margin: 0;
        color: white;
    }
    
    .dashboard-subtitle {
        font-size: 0.95rem;
        margin: 0.5rem 0 0 0;
        opacity: 0.9;
        color: #e0e7ff;
    }
    
    /* Stats cards */
    .stat-card {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.08);
        border-left: 4px solid;
        margin-bottom: 1rem;
    }
    
    .stat-card.total {
        border-left-color: #3b82f6;
    }
    
    .stat-card.encroachment {
        border-left-color: #dc2626;
    }
    
    .stat-card.vacant {
        border-left-color: #f59e0b;
    }
    
    .stat-card.compliant {
        border-left-color: #059669;
    }
    
    .stat-value {
        font-size: 2.5rem;
        font-weight: 700;
        margin: 0;
        line-height: 1;
    }
    
    .stat-label {
        font-size: 0.875rem;
        color: #64748b;
        margin-top: 0.5rem;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    /* Alerts table */
    .alert-high {
        background-color: #fee2e2;
        color: #991b1b;
        padding: 0.25rem 0.5rem;
        border-radius: 4px;
        font-weight: 600;
    }
    
    .alert-medium {
        background-color: #fef3c7;
        color: #92400e;
        padding: 0.25rem 0.5rem;
        border-radius: 4px;
        font-weight: 600;
    }
    
    /* Sidebar */
    .css-1d391kg {
        background-color: #1e293b;
    }
    
    /* Info box */
    .info-box {
        background: #eff6ff;
        border-left: 4px solid #3b82f6;
        padding: 1rem;
        border-radius: 6px;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Load GeoJSON data
@st.cache_data
def load_geojson(filepath):
    """Load GeoJSON file"""
    try:
        with open(filepath, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        st.error(f"File not found: {filepath}")
        return None

@st.cache_data
def load_geodataframe(filepath):
    """Load GeoJSON as GeoDataFrame"""
    try:
        return gpd.read_file(filepath)
    except Exception as e:
        st.error(f"Error loading GeoDataFrame: {e}")
        return None

def get_status_color(status):
    """Return color based on plot status"""
    colors = {
        'compliant': '#059669',  # Green
        'encroachment': '#dc2626',  # Red
        'vacant': '#f59e0b'  # Yellow/Orange
    }
    return colors.get(status.lower(), '#6b7280')

def calculate_statistics(plots_data, results_data):
    """Calculate dashboard statistics"""
    if not plots_data or not results_data:
        return {
            'total': 0,
            'compliant': 0,
            'encroachment': 0,
            'vacant': 0,
            'compliance_pct': 0
        }
    
    results_features = results_data['features']
    total = len(results_features)
    
    status_counts = {}
    for feature in results_features:
        status = feature['properties'].get('status', 'unknown')
        status_counts[status] = status_counts.get(status, 0) + 1
    
    compliant = status_counts.get('compliant', 0)
    encroachment = status_counts.get('encroachment', 0)
    vacant = status_counts.get('vacant', 0)
    
    compliance_pct = (compliant / total * 100) if total > 0 else 0
    
    return {
        'total': total,
        'compliant': compliant,
        'encroachment': encroachment,
        'vacant': vacant,
        'compliance_pct': round(compliance_pct, 1)
    }

def create_map(plots_data, results_data, status_filter='all'):
    """Create Folium map with plot overlays"""
    # Center on Nava Raipur, Chhattisgarh
    center = [21.16, 81.78]
    
    # Create map with CGGIS-like base layer
    m = folium.Map(
        location=center,
        zoom_start=13,
        tiles=None
    )
    
    # Add base layers
    folium.TileLayer(
        tiles='https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
        attr='Esri',
        name='Satellite',
        overlay=False,
        control=True
    ).add_to(m)
    
    folium.TileLayer(
        tiles='OpenStreetMap',
        name='Street Map',
        overlay=False,
        control=True
    ).add_to(m)
    
    # Add CSIDC reference marker
    folium.Marker(
        location=center,
        popup=folium.Popup(
            '<div style="font-family: Arial; width: 200px;">'
            '<h4 style="margin: 0 0 8px 0;">CSIDC GIS Portal</h4>'
            '<p style="margin: 0 0 8px 0; font-size: 12px;">Chhattisgarh State Industrial Development Corporation</p>'
            '<a href="https://cggis.cgstate.gov.in/csidc/" target="_blank" style="color: #3b82f6;">Visit Portal →</a>'
            '</div>',
            max_width=250
        ),
        tooltip='CSIDC GIS Portal Reference',
        icon=folium.Icon(color='orange', icon='industry', prefix='fa')
    ).add_to(m)
    
    # Add plot polygons from results
    if results_data:
        for feature in results_data['features']:
            props = feature['properties']
            status = props.get('status', 'unknown')
            
            # Apply filter
            if status_filter != 'all' and status != status_filter:
                continue
            
            plot_id = props.get('plot_id', 'N/A')
            compliance_score = props.get('compliance_score', 'N/A')
            violation_type = props.get('violation_type', 'None')
            severity = props.get('severity', 'N/A')
            
            # Get corresponding plot data
            plot_info = None
            if plots_data:
                for plot_feature in plots_data['features']:
                    if plot_feature['properties']['plot_id'] == plot_id:
                        plot_info = plot_feature['properties']
                        break
            
            # Create popup content
            popup_html = f"""
            <div style="font-family: Arial; width: 250px;">
                <h4 style="margin: 0 0 10px 0; color: {get_status_color(status)};">{plot_id}</h4>
                <table style="width: 100%; font-size: 12px;">
                    <tr><td><b>Owner:</b></td><td>{plot_info.get('owner', 'N/A') if plot_info else 'N/A'}</td></tr>
                    <tr><td><b>Area:</b></td><td>{plot_info.get('area', 'N/A') if plot_info else 'N/A'} sq.m</td></tr>
                    <tr><td><b>Zone:</b></td><td>{plot_info.get('zone', 'N/A') if plot_info else 'N/A'}</td></tr>
                    <tr><td><b>Status:</b></td><td><span style="color: {get_status_color(status)}; font-weight: bold;">{status.upper()}</span></td></tr>
                    <tr><td><b>Compliance:</b></td><td>{compliance_score}%</td></tr>
                    <tr><td><b>Last Updated:</b></td><td>{plot_info.get('last_updated', 'N/A') if plot_info else 'N/A'}</td></tr>
                </table>
                {f'<p style="margin-top: 10px; padding: 8px; background: #fee2e2; border-radius: 4px; font-size: 11px;"><b>Issue:</b> {plot_info.get("issue", violation_type)}</p>' if status != 'compliant' and plot_info else ''}
            </div>
            """
            
            # Add polygon
            folium.GeoJson(
                feature,
                style_function=lambda x, status=status: {
                    'fillColor': get_status_color(status),
                    'color': get_status_color(status),
                    'weight': 2,
                    'fillOpacity': 0.4
                },
                popup=folium.Popup(popup_html, max_width=300),
                tooltip=f"{plot_id} - {status.upper()}"
            ).add_to(m)
    
    # Add layer control
    folium.LayerControl().add_to(m)
    
    # Add fullscreen button
    plugins.Fullscreen().add_to(m)
    
    return m

def create_alerts_table(plots_data, results_data):
    """Create alerts table for problematic plots"""
    alerts = []
    
    if results_data:
        for feature in results_data['features']:
            props = feature['properties']
            status = props.get('status', 'unknown')
            
            if status in ['encroachment', 'vacant']:
                plot_id = props.get('plot_id', 'N/A')
                
                # Get plot info
                plot_info = None
                if plots_data:
                    for plot_feature in plots_data['features']:
                        if plot_feature['properties']['plot_id'] == plot_id:
                            plot_info = plot_feature['properties']
                            break
                
                issue_type = props.get('violation_type', plot_info.get('issue', 'Unknown') if plot_info else 'Unknown')
                severity = props.get('severity', 'medium')
                
                alerts.append({
                    'Plot ID': plot_id,
                    'Issue Type': issue_type.replace('_', ' ').title(),
                    'Severity': severity.upper(),
                    'Status': status.upper()
                })
    
    return pd.DataFrame(alerts)

# Main application
def main():
    # Header
    st.markdown("""
    <div class="dashboard-header">
        <h1 class="dashboard-title">🏭 Industrial Land Monitoring System</h1>
        <p class="dashboard-subtitle">CSIDC - Chhattisgarh State Industrial Development Corporation</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.image("https://via.placeholder.com/200x80/1e3a8a/ffffff?text=CSIDC", use_container_width=True)
        st.markdown("### 🗺️ Map Controls")
        
        # Status filter
        status_filter = st.selectbox(
            "Filter by Status",
            options=['all', 'compliant', 'encroachment', 'vacant'],
            format_func=lambda x: x.replace('_', ' ').title()
        )
        
        st.markdown("---")
        st.markdown("### 📤 Data Management")
        
        # File upload
        uploaded_file = st.file_uploader(
            "Upload New GeoJSON",
            type=['geojson', 'json'],
            help="Upload a new results.geojson file to update the map"
        )
        
        if uploaded_file:
            try:
                new_data = json.load(uploaded_file)
                st.success("✓ File uploaded successfully!")
                st.json(new_data, expanded=False)
            except Exception as e:
                st.error(f"Error loading file: {e}")
        
        st.markdown("---")
        st.markdown("### ℹ️ About")
        st.markdown("""
        <div class="info-box">
        This dashboard monitors industrial plot boundaries and detects violations by comparing official GIS data with inspection results.
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("**Reference:**")
        st.markdown("[CGGIS CSIDC Portal](https://cggis.cgstate.gov.in/csidc/)")
    
    # Load data
    plots_path = Path("data/plots.geojson")
    results_path = Path("data/results.geojson")
    
    plots_data = load_geojson(plots_path)
    results_data = load_geojson(results_path)
    
    if uploaded_file:
        results_data = new_data
    
    # Calculate statistics
    stats = calculate_statistics(plots_data, results_data)
    
    # Statistics cards
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div class="stat-card total">
            <div class="stat-value">{stats['total']}</div>
            <div class="stat-label">Total Plots</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="stat-card compliant">
            <div class="stat-value">{stats['compliant']}</div>
            <div class="stat-label">Compliant</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="stat-card encroachment">
            <div class="stat-value">{stats['encroachment']}</div>
            <div class="stat-label">Encroachments</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown(f"""
        <div class="stat-card vacant">
            <div class="stat-value">{stats['vacant']}</div>
            <div class="stat-label">Vacant Plots</div>
        </div>
        """, unsafe_allow_html=True)
    
    # Compliance percentage
    st.markdown(f"""
    <div style="background: white; padding: 1rem; border-radius: 10px; margin: 1rem 0; text-align: center;">
        <h3 style="margin: 0; color: #059669;">Compliance Rate: {stats['compliance_pct']}%</h3>
    </div>
    """, unsafe_allow_html=True)
    
    # Main content tabs
    tab1, tab2 = st.tabs(["🗺️ Interactive Map", "⚠️ Alerts & Violations"])
    
    with tab1:
        st.markdown("### Interactive Plot Map")
        st.markdown("*Click on any plot to view detailed information*")
        
        # Create and display map
        map_obj = create_map(plots_data, results_data, status_filter)
        st_folium(map_obj, width=None, height=600)
        
        # Legend
        st.markdown("""
        **Legend:**
        - 🟢 **Green**: Compliant plots
        - 🔴 **Red**: Encroachment detected
        - 🟡 **Yellow**: Vacant/Unused plots
        - 🟠 **Orange Marker**: CSIDC GIS Portal Reference
        """)
    
    with tab2:
        st.markdown("### Alerts & Violations")
        
        alerts_df = create_alerts_table(plots_data, results_data)
        
        if not alerts_df.empty:
            # Style the dataframe
            def highlight_severity(val):
                if val == 'HIGH':
                    return 'background-color: #fee2e2; color: #991b1b; font-weight: bold'
                elif val == 'MEDIUM':
                    return 'background-color: #fef3c7; color: #92400e; font-weight: bold'
                return ''
            
            styled_df = alerts_df.style.applymap(
                highlight_severity,
                subset=['Severity']
            )
            
            st.dataframe(styled_df, use_container_width=True, height=400)
            
            # Download button
            csv = alerts_df.to_csv(index=False)
            st.download_button(
                label="📥 Download Alerts Report (CSV)",
                data=csv,
                file_name=f"alerts_report_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv"
            )
        else:
            st.success("✓ No violations detected. All plots are compliant!")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #64748b; font-size: 0.875rem;">
        <p>Industrial Land Monitoring System | CSIDC | Government of Chhattisgarh</p>
        <p>Last Updated: {}</p>
    </div>
    """.format(datetime.now().strftime("%B %d, %Y at %I:%M %p")), unsafe_allow_html=True)

if __name__ == "__main__":
    main()
