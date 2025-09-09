"""
Jaipur Pincode Map Streamlit App

Features:
- Upload all-India pincode CSV (with lat/lon) OR rely on uploaded GeoJSON for pincode polygons
- Upload schools Excel (with Pincode, School Name, Board)
- Optional upload of Jaipur pincode GeoJSON/shapefile (GeoJSON recommended)
- Generates polygon map (uses GeoJSON polygons if provided, otherwise Voronoi approx from points)
- Shows permanent labels per-pincode: Pincode, School count, Boards
- Provides option to download the generated map HTML

How to run:
1. Create a virtual env (optional) and install requirements:
   pip install -r requirements.txt

requirements.txt (suggested):
streamlit
pandas
geopandas
folium
streamlit-folium
shapely
scipy
numpy
pyproj
fiona
rtree

2. Run:
   streamlit run jaipur_streamlit_map_app.py

Notes:
- On Streamlit Cloud, you must commit this file and requirements to a GitHub repo and connect the app.
- For accurate pincode boundaries, upload an official GeoJSON. Voronoi polygons are approximate.
"""

import io
import os
import tempfile
import webbrowser
import warnings

import streamlit as st
import pandas as pd
import geopandas as gpd
import folium
from shapely.geometry import Point, Polygon
from shapely.ops import unary_union
import numpy as np
from scipy.spatial import Voronoi
from streamlit_folium import st_folium

warnings.filterwarnings("ignore")

# ----------------- Utility functions (adapted from previous script) -----------------

def detect_pincode_col(df):
    for c in df.columns:
        if 'pin' in c.lower():
            return c
    raise ValueError('No pincode-like column found')

def detect_latlon_cols(df):
    lat_candidates = [c for c in df.columns if 'lat' in c.lower()]
    lon_candidates = [c for c in df.columns if any(k in c.lower() for k in ['lon','lng','long'])]
    if lat_candidates and lon_candidates:
        return lat_candidates[0], lon_candidates[0]
    # try common names
    for c in df.columns:
        if c.lower() in ('latitude','lat'):
            lat = c
        if c.lower() in ('longitude','lon','lng'):
            lon = c
    if 'lat' in locals() and 'lon' in locals():
        return lat, lon
    raise ValueError('Could not detect latitude/longitude columns. Found: ' + ', '.join(df.columns))


def create_geodf_from_points(df, latcol, loncol):
    df = df.dropna(subset=[latcol, loncol]).copy()
    geometry = [Point(xy) for xy in zip(df[loncol].astype(float), df[latcol].astype(float))]
    gdf = gpd.GeoDataFrame(df, geometry=geometry, crs='EPSG:4326')
    return gdf


def create_voronoi_polygons(gdf_points):
    gdf_proj = gdf_points.to_crs(epsg=3857)
    pts = np.array([[geom.x, geom.y] for geom in gdf_proj.geometry])
    if len(pts) < 3:
        raise ValueError('Need at least 3 points for Voronoi')
    vor = Voronoi(pts)
    regions = []
    for region_idx in vor.point_region:
        region = vor.regions[region_idx]
        if not region or -1 in region:
            regions.append(None); continue
        poly_pts = [vor.vertices[i] for i in region]
        regions.append(Polygon(poly_pts))
    polys = []
    for i in range(len(gdf_proj)):
        polys.append(regions[i])
    vor_gdf = gdf_proj.copy()
    vor_gdf['vor_poly'] = polys
    vor_gdf = vor_gdf[vor_gdf['vor_poly'].notnull()].copy()
    polygons = gpd.GeoDataFrame(vor_gdf.drop(columns=['geometry']), geometry='vor_poly', crs=gdf_proj.crs)
    hull = unary_union(gdf_proj.geometry).convex_hull
    polygons['geometry'] = polygons.geometry.intersection(hull)
    polygons = polygons.to_crs(epsg=4326)
    return polygons


def truncate_text(s, length=60):
    s = str(s)
    return s if len(s) <= length else s[:length-3] + '...'


# ----------------- Streamlit UI -----------------

st.set_page_config(layout='wide', page_title='Jaipur Pincode Schools Map')
st.title('Jaipur Pincode Map — Schools by Pincode')

with st.sidebar:
    st.header('Upload data')
    pincode_csv = st.file_uploader('Upload all-India pincode CSV (with lat,lon columns)', type=['csv'])
    polygons_file = st.file_uploader('Optional: Upload Jaipur pincode GeoJSON / Shapefile (.geojson .json .shp .zip)', type=['geojson','json','shp','zip'])
    schools_file = st.file_uploader('Upload Schools Excel (xlsx)', type=['xlsx','xls','csv'])
    st.markdown('---')
    st.write('Map options')
    show_points = st.checkbox('Show point markers', value=True)
    add_legend = st.checkbox('Add choropleth legend', value=False)
    use_voronoi = st.checkbox('Force Voronoi (even if polygons uploaded)', value=False)
    st.markdown('---')
    run_btn = st.button('Generate Map')

col1, col2 = st.columns([2,1])

with col2:
    st.info('Instructions:\n\n1) Upload all-India pincode CSV (with lat/lon).\n2) Upload schools Excel with columns including Pincode and Board.\n3) Optionally upload GeoJSON for accurate polygons.\n4) Click "Generate Map".')

# ----------------- Main logic -----------------

def process_files(pincode_csv_file, polygons_file_obj, schools_file_obj, force_voronoi=False):
    # Read pincode CSV
    if pincode_csv_file is None:
        st.warning('Please upload pincode CSV.')
        return None
    try:
        pincode_df = pd.read_csv(pincode_csv_file)
    except Exception as e:
        st.error('Failed to read pincode CSV: ' + str(e))
        return None

    try:
        latcol, loncol = detect_latlon_cols(pincode_df)
    except Exception as e:
        st.error('Could not detect lat/lon columns in pincode CSV: ' + str(e))
        return None

    try:
        pincode_col = detect_pincode_col(pincode_df)
    except Exception as e:
        st.error('Could not detect pincode column in pincode CSV: ' + str(e))
        return None

    pincode_df[pincode_col] = pincode_df[pincode_col].astype(str).str.strip()

    # Filter Jaipur by bounding box around Jaipur or ask user? We'll filter by district if exists, else by pincodes containing '302' or '303'
    if 'district' in [c.lower() for c in pincode_df.columns]:
        # find actual column name
        dcol = [c for c in pincode_df.columns if c.lower() == 'district'][0]
        pincode_df = pincode_df[pincode_df[dcol].astype(str).str.contains('jaipur', case=False, na=False)]
    else:
        # simple heuristic: keep pincodes starting with '302' or '303' (Jaipur area)
        pincode_df = pincode_df[pincode_df[pincode_col].str.startswith(('302','303'))]

    if pincode_df.empty:
        st.error('No Jaipur rows found in pincode CSV after filtering. Please ensure CSV has Jaipur data or upload polygons file.')
        return None

    gdf_points = create_geodf_from_points(pincode_df, latcol, loncol)

    # Read schools file
    if schools_file_obj is None:
        st.error('Please upload schools file (xlsx/csv).')
        return None
    try:
        if schools_file_obj.name.lower().endswith('.csv'):
            schools_df = pd.read_csv(schools_file_obj)
        else:
            schools_df = pd.read_excel(schools_file_obj)
    except Exception as e:
        st.error('Failed to read schools file: ' + str(e))
        return None

    # detect columns in schools
    school_pcol = detect_pincode_col(schools_df)
    schools_df[school_pcol] = schools_df[school_pcol].astype(str).str.strip()
    school_name_col = next((c for c in schools_df.columns if 'school' in c.lower()), schools_df.columns[0])
    board_col = next((c for c in schools_df.columns if 'board' in c.lower()), None)
    if board_col is None:
        schools_df['Board'] = ''
        board_col = 'Board'

    # aggregate
    school_stats = schools_df.groupby(school_pcol).agg(
        SchoolCount=(school_name_col, 'count'),
        Boards=(board_col, lambda x: ', '.join(sorted({str(v).strip() for v in x if pd.notna(v) and str(v).strip()})))
    ).reset_index()

    # If user provided polygons and not forcing voronoi
    polygons = None
    if polygons_file_obj is not None and not force_voronoi:
        # try read geojson or zip/shapefile
        try:
            # save uploaded file to temp and read with geopandas
            tmp = tempfile.NamedTemporaryFile(delete=False)
            tmp.write(polygons_file_obj.getvalue())
            tmp.flush()
            tmp.close()
            polygons = gpd.read_file(tmp.name)
            os.unlink(tmp.name)
        except Exception as e:
            st.warning('Could not read uploaded polygons file, will fallback to Voronoi. Error: ' + str(e))
            polygons = None

    if polygons is None:
        st.info('Creating Voronoi polygons (approximate) from pincode points...')
        try:
            polygons = create_voronoi_polygons(gdf_points)
        except Exception as e:
            st.error('Voronoi creation failed: ' + str(e))
            return None

    # detect pincode col in polygons
    try:
        poly_pcol = detect_pincode_col(polygons)
    except Exception:
        # try common names
        poly_pcol = None
        for c in polygons.columns:
            if 'pin' in c.lower():
                poly_pcol = c
                break
    if poly_pcol is None:
        # we will try to map by nearest point pincode
        polygons['__idx__'] = range(len(polygons))
        # not ideal — but try to transfer pincode via spatial join with points
        gdf_points = gdf_points.reset_index().rename(columns={'index': 'pt_idx'})
        joined = gpd.sjoin_nearest(polygons.set_geometry('geometry'), gdf_points.set_geometry('geometry'), how='left', distance_col='dist')
        if 'pt_idx' in joined.columns:
            polygons['Pincode'] = joined['pincode'] if 'pincode' in gdf_points.columns else gdf_points.columns[0]
            poly_pcol = 'Pincode'
        else:
            poly_pcol = None

    # merge polygon with school stats
    if poly_pcol and poly_pcol in polygons.columns:
        polygons[poly_pcol] = polygons[poly_pcol].astype(str).str.strip()
        merged = polygons.merge(school_stats, how='left', left_on=poly_pcol, right_on=school_pcol)
    else:
        # fallback: no pincode in polygons; create numeric index join
        merged = polygons.copy()
        merged['SchoolCount'] = 0
        merged['Boards'] = ''

    merged['SchoolCount'] = merged.get('SchoolCount', 0).fillna(0).astype(int)
    merged['Boards'] = merged.get('Boards', '').fillna('').astype(str)

    return {
        'map_gdf': merged,
        'points_gdf': gdf_points,
        'pcode_col': poly_pcol or detect_pincode_col(gdf_points)
    }


# ----------------- Map rendering -----------------

def render_map(map_gdf, points_gdf, pcode_col, show_points=True):
    m = folium.Map(location=[26.9124, 75.7873], zoom_start=11)
    max_count = int(map_gdf['SchoolCount'].max() or 1)

    def style_func(feature):
        count = feature['properties'].get('SchoolCount', 0) or 0
        norm = min(1.0, count / max_count) if max_count > 0 else 0
        r = int(255 * norm)
        g = int(180 * (1 - norm))
        b = 80
        return {
            'fillColor': f'#{r:02x}{g:02x}{b:02x}',
            'color': 'black', 'weight': 1, 'fillOpacity': 0.6
        }

    folium.GeoJson(map_gdf, style_function=style_func).add_to(m)

    # add centroid labels
    for _, row in map_gdf.iterrows():
        geom = row.geometry
        if geom is None or geom.is_empty:
            continue
        centroid = geom.centroid
        boards = truncate_text(row.get('Boards', ''), length=60)
        label_html = (
            f"<div style='text-align:center; font-size:11px; line-height:1.1;'>"
            f"<b>{row.get(pcode_col,'')}</b><br>Schools: {int(row.get('SchoolCount',0))}<br>{boards}</div>"
        )
        folium.Marker(location=[centroid.y, centroid.x], icon=folium.DivIcon(html=label_html)).add_to(m)

    if show_points:
        for _, r in points_gdf.iterrows():
            folium.CircleMarker(location=[r.geometry.y, r.geometry.x], radius=2, fill=True).add_to(m)

    return m


# ----------------- Button action -----------------

if run_btn:
    st.info('Processing files... this may take a few seconds')
    result = process_files(pincode_csv, polygons_file, schools_file, force_voronoi=use_voronoi)
    if result is None:
        st.error('Processing failed. Check messages above.')
    else:
        map_gdf = result['map_gdf']
        pts = result['points_gdf']
        pcol = result['pcode_col']

        st.success('Data processed. Rendering map...')
        m = render_map(map_gdf, pts, pcol, show_points=show_points)

        # display in Streamlit
        st_data = st_folium(m, width=900, height=700)

        # provide download link for HTML
        tmpdir = tempfile.mkdtemp()
        out_path = os.path.join(tmpdir, 'jaipur_pincode_map.html')
        m.save(out_path)
        with open(out_path, 'rb') as f:
            btn = st.download_button('Download map HTML', data=f, file_name='jaipur_pincode_map.html', mime='text/html')

        st.info('Map generation finished.')

else:
    st.write('Upload files and click "Generate Map"')
