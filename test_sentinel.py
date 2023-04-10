from sentinelhub import WebFeatureService, BBox, CRS

# Define the latitude and longitude coordinates for the area of interest
lat, lon = 12.34, 56.78

# Create a bounding box object for the area of interest
bbox = BBox(bbox=[lon-0.01, lat-0.01, lon+0.01, lat+0.01], crs=CRS.WGS84)

# Create a Web Feature Service object to query Sentinel Hub for imagery
INSTANCE_ID = "0d040e26-c8a8-4562-a4bc-fc94506009ea"
wfs = WebFeatureService(instance_id=INSTANCE_ID, 
                        data_source='S2L1C', 
                        bbox=bbox, 
                        time=('2022-01-01', '2022-01-31'))

# Download the imagery for the specified area and time period
images = wfs.get_data()