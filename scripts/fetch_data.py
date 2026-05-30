import datetime as dt
import logging
import os
from pathlib import Path

from dotenv import load_dotenv
from sentinelhub import (
    CRS,  #coordinate reference system
    BBox,  #define a bounding box
    DataCollection,  #Specify the satellite dataset you want
    Geometry,  #can represet a point, line, or polygon
    MimeType,  #specify the format of the data you want
    SentinelHubCatalog,  #for searching and retrieving data
    SentinelHubRequest,  #for making requests to Sentinel Hub
    SHConfig,  #configuration for Sentinel Hub
    bbox_to_dimensions,  #convert a bounding box to dimensions
)
from tifffile import imwrite

load_dotenv()


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

CLIENT_ID = os.getenv('SH_CLIENT_ID')
CLIENT_SECRET = os.getenv('SH_CLIENT_SECRET')

config = SHConfig()

if CLIENT_ID and CLIENT_SECRET:
    config.sh_client_id = CLIENT_ID
    config.sh_client_secret = CLIENT_SECRET
    logging.info("Sentinel Hub configuration set with client ID and secret.")
else:
    logging.error("Sentinel Hub client ID and secret are not set. Please check your .env file.")
    raise ValueError("Sentinel Hub client ID and secret are not set.")

config.sh_base_url = "https://sh.dataspace.copernicus.eu"
config.sh_token_url = "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token"


WORKDIR = os.environ.get('WORKDIR', str(Path.cwd()))
WORKDIR = Path(WORKDIR)

OUTPUT_DIR = WORKDIR / 'data' / 'raw'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
logging.info(f"Output directory set to: {OUTPUT_DIR}")

AOI_BBOX = [6.9, 50.5, 7.3, 50.6]  # Ahr Valley, Germany
AOI_CRS = CRS.WGS84  # these values are in degrees, using WGS84
TIME_INTERVAL = ('2021-07-01', '2021-07-25')
RESOLUTION = 10  # in meters


evalscript_s1 = """
//VERSION=3
function setup() {
    return {
        input: ["VV", "VH"],
        output: { bands: 2, sampleType: "FLOAT32" }
    };
}
function evaluatePixel(sample) {
    return [sample.VV, sample.VH];
}
"""

evalscript_s2 = """
//VERSION=3
function setup() {
  return {
    input: ["B02", "B03", "B04", "B08", "SCL", "dataMask"],
    output: { bands: 5, sampleType: "UINT16" }
  };
}
function evaluatePixel(sample) {
  if (sample.dataMask == 0) {
      return [0, 0, 0, 0, 0];
  }
  let cloudMask = 0;
  if (sample.SCL == 3 || sample.SCL == 8 || sample.SCL == 9 || sample.SCL == 10) {
      cloudMask = 1;
  }
  return [sample.B02, sample.B03, sample.B04, sample.B08, cloudMask];
}
"""

def search_available_scenes(catalog: SentinelHubCatalog, data_collection: DataCollection, geometry: Geometry, time_interval: tuple) -> list[str]:
    """
    Search for available scenes in the specified area and time interval.
    
    :param catalog: SentinelHubCatalog instance
    :param data_collection: DataCollection to search in
    :param geometry: Geometry of the area of interest
    :param time_interval: Tuple with start and end date
    :return: List of scene IDs
    """
    logging.info(f"Searching for scenes in {data_collection.name} from {time_interval[0]} to {time_interval[1]}")

    search_iterator = catalog.search(
        collection=data_collection,
        geometry=geometry,
        time=time_interval,
        fields={"include": ["properties.datetime"], "exclude": []},
    )
    timestamps = [item["properties"]["datetime"] for item in search_iterator]
    logging.info(f"Found {len(timestamps)} scenes.")
    return sorted(list(set(timestamps)))  # Return unique timestamps sorted

def download_scenes(timestamps: list, data_collection:DataCollection, evalscript: str, file_prefix: str):
    """
    Download scenes for the given timestamps from Sentinel Hub.
    :param timestamps: List of timestamps to download
    :param data_collection: DataCollection to download from
    :param evalscript: Evalscript for processing the data
    :param file_prefix: Prefix for the output files
    """
    if not timestamps:
        logging.warning("No timestamps provided for downloading scenes.")
        return

    logging.info(f"Downloading {len(timestamps)} scenes from {data_collection.name}... with prefix '{file_prefix}'")
    aoi_bbox_obj = BBox(AOI_BBOX, crs=AOI_CRS)
    size = bbox_to_dimensions(aoi_bbox_obj, resolution=RESOLUTION)

    MAX_SIZE = 2500 # Sentinel Hub limits the size of the request
    if size[0] > MAX_SIZE or size[1] > MAX_SIZE:
        scale = min(MAX_SIZE / size[0], MAX_SIZE / size[1])
        size = (int(size[0] * scale), int(size[1] * scale))
        logging.warning(f"Resized dimensions to fit API limits: {size}")

    for ts in timestamps:
        acquisition_time = dt.datetime.fromisoformat(ts.replace('Z', '+00:00')) # Convert to datetime object
        time_slot = (
            acquisition_time - dt.timedelta(minutes=30),
            acquisition_time + dt.timedelta(minutes=30)
        )
        logging.debug(f"Processing scene for time slot: {time_slot}")

        request = SentinelHubRequest(
            evalscript=evalscript,
            input_data=[
                SentinelHubRequest.input_data(
                    data_collection=data_collection,
                    time_interval=time_slot,
                )
            ],
            responses=[
                SentinelHubRequest.output_response(
                    "default", MimeType.TIFF                )
            ],
            bbox=aoi_bbox_obj,
            size=size,
            config=config,
        )

        try:
            data = request.get_data()[0]
            filename_ts = acquisition_time.strftime("%Y%m%dT%H%M%S")
            filename = OUTPUT_DIR / f"{file_prefix}_{filename_ts}.tiff"
            imwrite(filename, data)
            logging.info(f"Downloaded scene for {acquisition_time} to {filename}")
        except Exception as e:
            logging.error(f"Failed to download scene for {acquisition_time}: {e}")


if __name__ == "__main__":
    aoi_geometry = Geometry(BBox(AOI_BBOX, crs=AOI_CRS).geometry, crs=AOI_CRS)
    catalog = SentinelHubCatalog(config=config)
    logging.info("Starting data fetch process...")

    # Search and download Sentinel-1 scenes
    S1_COLLECTION = DataCollection.define(
        name = "SENTINEL1_GRD_CDSE",
        api_id = "sentinel-1-grd"
    )
    S2_COLLECTION = DataCollection.define(
        name = "SENTINEL2_L2A_CDSE",
        api_id = "sentinel-2-l2a"
    )
    #S1 - Sentinel-1 GRD (Ground Range Detected) SAR radar data
    s1_timestamps = search_available_scenes(catalog, S1_COLLECTION, aoi_geometry, TIME_INTERVAL)
    #S2 - Sentinel-2 L2A (Level 2A) optical data with cloud masking
    s2_timestamps = search_available_scenes(catalog, S2_COLLECTION, aoi_geometry, TIME_INTERVAL)

    logging.info("-" * 50)
    if s1_timestamps: logging.info(f"Found {len(s1_timestamps)} available Sentinel-1 scenes.")
    else: logging.warning("No Sentinel-1 data found.")
    if s2_timestamps: logging.info(f"Found {len(s2_timestamps)} available Sentinel-2 scenes.")
    else: logging.warning("No Sentinel-2 data found.")
    logging.info("-" * 50)

    download_scenes(s1_timestamps, S1_COLLECTION, evalscript_s1, "s1_iw_grd")
    download_scenes(s2_timestamps, S2_COLLECTION, evalscript_s2, "s2_l2a")

    logging.info("\nData acquisition process complete.")
