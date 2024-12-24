"""
Descarga de datos de relieve
"""
# ---------------------------------------------------------------------------- #
import xarray as xr
import numpy as np
# ---------------------------------------------------------------------------- #

def compute():

    try:
        topografia = xr.open_dataset('~/PronoClim/topografia_sa.nc')
    except:
        print('Datos topografia no disponibles')
        print('Descargando...')

        # la lat y lon del modelo
        data = xr.open_dataset(
            '~/PronoClim/modelos_seteados/prec_CMC-CanCM4i-IC3_SON.nc')

        # -------------------------------------------------------------------- #
        # OPCION 1:
        # -------------------------------------------------------------------- #
        # Necesita la libreria rasterio. pip install rasterio (con el entorno de conda
        # activado)
        import os
        try:
            import rasterio
            check = True
        except:
            print('Error: rasterio no instalado')
            print('con el entorno de conda activado --> "pip install rasterio"')
            check = False

        lon_max = data.lon.values.max() - 360
        lon_min = data.lon.values.min() - 360
        lat_max = data.lat.values.max()
        lat_min = data.lat.values.min()

        # Datos de https://www.ncei.noaa.gov/products/etopo-global-relief-model
        # Idea inspirada en el funcionamiento de MetR
        # https://eliocamp.github.io/metR/reference/GetTopography.html

        url = ('https://gis.ngdc.noaa.gov/arcgis/rest/services/DEM_mosaics/'
               'DEM_all/ImageServer/exportImage?bbox=' + str(lon_min) + ','
               + str(lat_min) + ',' + str(lon_max) + ',' + str(lat_max) +
               '&bboxSR=4326&size=4123,4472&imageSR=4326&format=tiff&pixelType='
               'F32&interpolation=+RSP_NearestNeighbor&compression=LZ77&'
               'renderingRule={%22rasterFunction%22:%22none%22}&mosaicRule='
               '{%22where%22:%22Name=%27ETOPO_2022_v1_60s_surface%27%22}&f=image')

        if check is True:
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

            topografia = xr.Dataset({"topo": data_array})

            topografia.to_netcdf('~/PronoClim/topografia_sa.nc')

    return topografia
# ---------------------------------------------------------------------------- #