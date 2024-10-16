"""
Descarga de datos de relieve
"""
# ---------------------------------------------------------------------------- #
import srtm # pip install srtm.py
import xarray as xr
import numpy as np
# ---------------------------------------------------------------------------- #
# la lat y lon del modelo
data = xr.open_dataset(
    '~/PronoClim/modelos_seteados/prec_CMC-CanCM4i-IC3_SON.nc')

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