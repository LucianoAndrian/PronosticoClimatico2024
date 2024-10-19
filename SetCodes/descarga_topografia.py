"""
Descarga de datos de relieve
"""
# ---------------------------------------------------------------------------- #
import xarray as xr
import numpy as np
# ---------------------------------------------------------------------------- #
# la lat y lon del modelo
data = xr.open_dataset(
    '~/PronoClim/modelos_seteados/prec_CMC-CanCM4i-IC3_SON.nc')

# ---------------------------------------------------------------------------- #
# OPCION 1:
# ---------------------------------------------------------------------------- #
# Necesita la libreria rasterio. pip install rasterio (con el entorno de conda
# activado)
import os
import rasterio

lon_max = data.lon.values.max() - 360
lon_min = data.lon.values.min() - 360
lat_max = data.lat.values.max()
lat_min = data.lat.values.min()

# Datos de https://www.ncei.noaa.gov/products/etopo-global-relief-model
# Idea inspirada en el funcionamiento de MetR
# https://eliocamp.github.io/metR/reference/GetTopography.html

url = ('https://gis.ngdc.noaa.gov/arcgis/rest/services/DEM_mosaics/DEM_all/'
       'ImageServer/exportImage?bbox=' + str(lon_min) + ',' + str(lat_min) +
       ',' + str(lon_max) + ',' + str(lat_max) + '&bboxSR=4326&size=4123,' 
       '4472&imageSR=4326&format=tiff&pixelType=F32&interpolation='
       '+RSP_NearestNeighbor&compression=LZ77&renderingRule='
       '{%22rasterFunction%22:%22none%22}&mosaicRule='
       '{%22where%22:%22Name=%27ETOPO_2022_v1_60s_surface%27%22}&f=image')

output_file = 'topography.tiff'
command = f'wget -O {output_file} "{url}"'
os.system(command)

with rasterio.open("topography.tiff") as dataset:
    topography_data = dataset.read(1)

    transform = dataset.transform
    lon, lat = np.meshgrid(
        np.arange(dataset.width) * transform[0] + transform[2],
        np.arange(dataset.height) * transform[4] + transform[5]
    )

data_array = xr.DataArray(
    topography_data,
    dims=["lat", "lon"],
    coords={"lat": lat[:, 0], "lon": lon[0, :]},
    attrs={"units": "meters", "description": "Topografía"}
)

dataset = xr.Dataset({"topo": data_array})

dataset.to_netcdf('~/PronoClim/topografia_sa_hr.nc')
# ---------------------------------------------------------------------------- #
# OPCION 2. MUY LENTO incluso usando 1x1ª
# ---------------------------------------------------------------------------- #
# Necesita la libraria srtm.py. pip install srtm.py
import srtm # pip install srtm.py
elevation_data = srtm.get_data()
lon, lat = np.meshgrid(data.lon.values - 360, data.lat.values)

def aux_get_elevation(lat, lon):
    elev = elevation_data.get_elevation(lat, lon)
    if elev is None:
        return 0
    return elev

# tarda aprox 30 min en descargar (desde la facu)
terreno = np.vectorize(aux_get_elevation)(lat, lon)

ds = xr.Dataset(
    {"topo": (["lat", "lon"], terreno)
     },
    coords={
        "lat": data.lat.values,
        "lon": data.lon.values

    }
)

ds.to_netcdf('~/PronoClim/topografia_sa.nc')
# ---------------------------------------------------------------------------- #