import os
import requests
import zipfile
import geopandas as gpd


def extract_northern_cape_roads():
    # 1. Configuration
    url = "https://download.geofabrik.de/africa/south-africa-latest-free.shp.zip"
    local_zip = "south-africa-latest-free.shp.zip"
    extract_dir = "temp_sa_roads"
    output_gpkg = "northern_cape_roads.gpkg"

    # 2. Download zip with a standard browser User-Agent (Bypasses bot-blocking)
    if not os.path.exists(local_zip) or not zipfile.is_zipfile(local_zip):
        print("📥 Downloading South Africa road data archive from Geofabrik...")

        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
        response = requests.get(url, headers=headers, stream=True)
        response.raise_for_status()
        if 'zip' not in response.headers.get('Content-Type', ''):
            raise ValueError(f"Expected a zip file, got Content-Type: {response.headers.get('Content-Type')}")

        with open(local_zip, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
        print("Download complete!")
    else:
        print("Found existing local file check. (If it fails again, delete this zip file manually).")

    # 3. Unzip files locally
    print("Unzipping target road vector layer components...")
    os.makedirs(extract_dir, exist_ok=True)

    target_prefixes = ["gis_osm_roads_free_1.shp", "gis_osm_roads_free_1.shx", "gis_osm_roads_free_1.dbf",
                       "gis_osm_roads_free_1.prj"]
    with zipfile.ZipFile(local_zip, 'r') as zip_ref:
        for file in zip_ref.namelist():
            if any(file.endswith(prefix) for prefix in target_prefixes):
                zip_ref.extract(file, extract_dir)

    # 4. Load from the unzipped directory
    shp_path = os.path.join(extract_dir, "gis_osm_roads_free_1.shp")
    print(f"Loading unzipped shapefile vector layers...")

    try:
        roads_gdf = gpd.read_file(shp_path, engine="pyogrio")
        print(f"Loaded {len(roads_gdf):,} total road segments across South Africa.")

        # 5. Fast Coordinate Slice for Northern Cape Province
        print("Filtering for Northern Cape boundaries...")
        min_lon, max_lon = 16.45, 25.30
        min_lat, max_lat = -31.85, -24.60
        nc_roads = roads_gdf.cx[min_lon:max_lon, min_lat:max_lat]

        # 6. Save to clean target GeoPackage format
        print(f"Saving {len(nc_roads):,} roads to {output_gpkg}...")
        nc_roads.to_file(output_gpkg, driver="GPKG", layer="edges", engine="pyogrio")

        # 7. Cleanup raw source files
        print("Cleaning up temporary source archives...")
        if os.path.exists(local_zip):
            os.remove(local_zip)
        if os.path.exists(extract_dir):
            for root, dirs, files in os.walk(extract_dir, topdown=False):
                for name in files:
                    os.remove(os.path.join(root, name))
                for name in dirs:
                    os.rmdir(os.path.join(root, name))
            os.rmdir(extract_dir)

        print("🎉 Success! Your spatial road dataset is complete.")

    except Exception as e:
        print(f"Operation failed: {e}")


if __name__ == "__main__":
    extract_northern_cape_roads()
