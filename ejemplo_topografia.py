"""
Ejemplo de uso de topografia para enmascarar relieve usando el archivo:
topografia_sa.nc disponible en repositorio de github.

Es facilmente adaptable a la funcion PlotContourf_SA
"""
# ---------------------------------------------------------------------------- #
import xarray as xr
import matplotlib.pyplot as plt
import numpy as np
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
import cartopy.crs as ccrs
# ---------------------------------------------------------------------------- #
# Cambiar rutas según corresponda
mod_cm4 =  xr.open_dataset(
    '~/PronoClim/modelos_seteados/prec_CMC-CanCM4i-IC3_SON.nc')
data = mod_cm4.sel(time='2005-08-01').mean('r')

topografia = xr.open_dataset('~/PronoClim/topografia_sa.nc')

# si se quiere una topografia con alta resolución
# se puede correr la opcion 1 de SetCodes/descarga_topografia.py
# necesita la libreria rasterio.
#topografia = xr.open_dataset('~/PronoClim/topografia_sa_hr.nc')


# figura
fig = plt.figure(figsize=(5,6), dpi=100)

crs_latlon = ccrs.PlateCarree()
ax = plt.axes(projection=ccrs.PlateCarree(central_longitude=180))
ax.set_extent([275, 330, -60, 20], crs=crs_latlon)

# Contornos
im = ax.contourf(data.lon,
                 data.lat,
                 data.prec[0,:,:],
                 levels=np.arange(0,15,1),
                 transform=crs_latlon, cmap='YlGnBu', extend='both')

# barra de colores
cb = plt.colorbar(im, fraction=0.042, pad=0.035, shrink=0.8)
cb.ax.tick_params(labelsize=8)

# ------------------------- Enmascarando relieve ----------------------------- #
from matplotlib import colors
cmap2 = colors.ListedColormap(['k']) # una palenta de colores todo negro

# contorno que va enmascarar el relieve superior a mask_level
# probar varios niveles
mask_level = 1300 #metros
ax.contourf(topografia.lon, topografia.lat, topografia.topo,
                levels=[mask_level,666666],
                cmap=cmap2, transform=crs_latlon)
# ---------------------------------------------------------------------------- #

ax.coastlines(color='k', linestyle='-', alpha=1)

ax.set_xticks(np.arange(275, 330, 10), crs=crs_latlon)
ax.set_yticks(np.arange(-60, 40, 20), crs=crs_latlon)
lon_formatter = LongitudeFormatter(zero_direction_label=True)
lat_formatter = LatitudeFormatter()
ax.xaxis.set_major_formatter(lon_formatter)
ax.yaxis.set_major_formatter(lat_formatter)
ax.gridlines(crs=crs_latlon, linewidth=0.3, linestyle='-')
ax.tick_params(labelsize=10)

plt.title('Ejemplo - topografia', fontsize=12)

plt.tight_layout()
plt.show()
# ---------------------------------------------------------------------------- #
# ---------------------------------------------------------------------------- #