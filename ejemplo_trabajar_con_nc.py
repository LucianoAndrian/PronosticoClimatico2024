"""
La manera mas facil y practica de abrir archivos .nc en python y para este curso
es utlizando la libreria xarray
"""
# ---------------------------------------------------------------------------- #
import xarray as xr
# ---------------------------------------------------------------------------- #
# abrir el archivo
ruta = '~/PronoClim/obs_seteadas/'
ds = xr.open_dataset(ruta + 'tref_monthly_nmme_ghcn_cams_sa.nc')

ds # ver metadata
ds.dims # ver dimensiones
ds.data_vars # ver variables

# ver mas metadata de la variable
ds['tref']
ds.tref # esto igual al anterior

# --- extraer variables y dimensiones --- #
temp = ds['tref']
temp = ds.tref

# temp es un DataArray.
# Estas variables pueden ser tratadas como ndarray de numpy:
tref_np = ds.tref.values
tref_np = temp.values

# Tambien se puede hacer con las dimensiones
lon = ds.lon
lon = ds['lon']

# ---------------------------------------------------------------------------- #
# Seleccion de dominios espaciales y/o temporales ---------------------------- #
# ---------------------------------------------------------------------------- #
ds_cut = ds.sel(time=slice('2003-01-01', '2016-12-01'),
                lon=slice(280, 290),
                lat=slice(-45, -20))
# CUIDADO, en este caso si fuera lat=slice(-20, -45) hubiese resultado en la
# dimension "lat" de "ds_cut" vacia y sin dar ningún mensaje de error

# ---------------------------------------------------------------------------- #
# Recursos practicos para seleccionar: --------------------------------------- #
# ---------------------------------------------------------------------------- #
# determinados años y todos los meses de esos años
anios = [1985, 1989, 2005, 2007]
ds_sel_anios = ds.sel(time=ds.time.dt.year.isin(anios))

# determinado/s mes/es para todos los años

ds_oct = ds.sel(time=ds.time.dt.month.isin(10)) # ej. octubre

ds_mjj = ds.sel(time=ds.time.dt.month.isin([5, 6, 7])) # ej. mayo, junio, julio

# seleccion de frecuencias por fechas
ds_anual_mean = ds.resample(time='YE').mean() # ej. promedios anuales
# estos ultimos 4 ejemplos solo funcionan con la dimension 'time' codificada
# con formato de fechas

# seleccionar desde una fecha en particular y el resto hacia atras o adelante
ds_2005_present = ds.sel(time=slice('2005-01-01', None)) # ej. 2005 -> presente

# ---------------------------------------------------------------------------- #
# Algunas cuentas ------------------------------------------------------------ #
# ---------------------------------------------------------------------------- #
ds_media = ds.mean() # media total (lon, lat, time -> promedio a un valor)
ds_std = ds.std() # desvio total

ds_media_temporal = ds.mean('time') # media temporal, devuelve un campo lon, lat
ds_std_temporal =  ds.std('time') # SD temporal, devuelve un campo lon, lat
ds_var_temporal = ds.var('time') # varianza ...

# --- pueden combinarse con las funciones de seleccion --- #
ds_media_temporal_mjj = ds.sel(
    time=ds.time.dt.month.isin([5, 6, 7])).mean('time')

ds_media_caja = ds.sel(
    lon=slice(285, 290), lat=slice(-45, -20)).mean(['lon', 'lat'])
# esto es una serie temporal del promedio del dominio seleccionado

# ---------------------------------------------------------------------------- #
# Percentiles para cada punto de reticula ------------------------------------ #
# ---------------------------------------------------------------------------- #
ds_q = ds.quantile(q=[0.05, 0.95], dim='time') # a lo largo de la dim=time
# ds_q es un campo de lon, lat, y quantile.

# ---------------------------------------------------------------------------- #
# Anomalías para cada punto de reticula -------------------------------------- #
# ---------------------------------------------------------------------------- #
# Anomalia temporal restando a todos los campos el campo medio
ds_anomalias_temporales = ds - ds_media_temporal

# Se pueden hacer otras operaciones entre estas variables, siempre y cuando
# el nombre y longitud de sus dimenciones y variables sean las mismas.
ds_anomalias_temporales_norm = (ds - ds_media_temporal)/ds_std_temporal

# Anomalias mensuales, filtrando el ciclo anual
# a cada mes de cada año le va restar la media de ese mes en todo el periodo
# esto solo es posible con la dimension 'time' con formato de fechas.
ds_anom_mon = ds.groupby('time.month') - ds.groupby('time.month').mean('time')

# ---------------------------------------------------------------------------- #
# Promedio movil para cada punto de reticula --------------------------------- #
# ---------------------------------------------------------------------------- #
# ej. promedio movil de 3 meses posicionando el valor promediado en el centro
# es decir, el promedio de mayo, junio, julio se va ubicar en junio de cada año
# en el ds resultante
ds_rm3 = ds.rolling(time=3, center=True).mean()

# ---------------------------------------------------------------------------- #
# Filtrar la tendencia para cada punto de reticula --------------------------- #
# ---------------------------------------------------------------------------- #
# 1.
# Ajusta una regresión lineal de la variable en ds con la dimencion 'time'
# Se ajusta a un polinomio de grado 1, una recta
# Se obtiene un campo de lon, lat y los coeficientes de la recta para
# cada punto de reticula
aux_coef = ds_anomalias_temporales_norm.polyfit(dim='time', deg=1, skipna=True)

# 2.
# Se evalua el polinomio ajustado en los pasos temporales para cada punto
# de reticula
# Se obtiene un campo lon, lat, time que contiene la recta de tendencia lineal
# para cada punto y tiempo.
ds_trend = xr.polyval(ds_anomalias_temporales_norm.time,
                      aux_coef.tref_polyfit_coefficients)

# 3.
# Resta la tendencia de los datos originales
ds_anom_norm_detrended = ds_anomalias_temporales_norm - ds_trend

# Podemos verlo en un punto en particular
import matplotlib.pyplot as plt
plt.plot(ds_anomalias_temporales_norm.sel(lon=300, lat=0).tref,
         label='original')
plt.plot(ds_anom_norm_detrended.sel(lon=300, lat=0).tref,
         label='Detrended')
plt.legend()
plt.grid()
plt.show()

# ---------------------------------------------------------------------------- #
plt.close('all')

# ---------------------------------------------------------------------------- #
# Ploteo de datos georeferenciados
# ---------------------------------------------------------------------------- #

# Se puede plotear directamente desde xarray --------------------------------- #
# Bueno para graficos exploratorios y rapidos
ds_media_temporal.tref.plot.pcolormesh(vmin=0, vmax=30, cmap='Spectral_r')
plt.show() # esto puede cambiar segun se haga por terminal o IDE

# ---------------------------------------------------------------------------- #
# Usando matplotlib.pyplot explicitamente ------------------------------------ #
# ---------------------------------------------------------------------------- #
import matplotlib.pyplot as plt
import numpy as np
import cartopy.feature
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
import cartopy.crs as ccrs

crs_latlon = ccrs.PlateCarree()

fig = plt.figure(figsize=(5,6), dpi=100)

ax = plt.axes(projection=ccrs.PlateCarree(central_longitude=180))
ax.set_extent([275, 330, -60, 20], crs=crs_latlon)

# Contornos
im = ax.contourf(ds_media_temporal.lon,
                 ds_media_temporal.lat,
                 ds_media_temporal.tref,
                 levels=np.arange(0,30),
                 transform=crs_latlon, cmap='Spectral_r', extend='both')

# barra de colores
cb = plt.colorbar(im, fraction=0.042, pad=0.035, shrink=0.8)
cb.ax.tick_params(labelsize=8)

# cosas varias del mapa, no tdo es necesario
ax.add_feature(cartopy.feature.LAND, facecolor='white', edgecolor='k')
ax.add_feature(cartopy.feature.BORDERS, linestyle='-', color='k')
ax.add_feature(cartopy.feature.STATES)
ax.coastlines(color='k', linestyle='-', alpha=1)

ax.set_xticks(np.arange(275, 330, 10), crs=crs_latlon)
ax.set_yticks(np.arange(-60, 40, 20), crs=crs_latlon)
lon_formatter = LongitudeFormatter(zero_direction_label=True)
lat_formatter = LatitudeFormatter()
ax.xaxis.set_major_formatter(lon_formatter)
ax.yaxis.set_major_formatter(lat_formatter)

ax.gridlines(crs=crs_latlon, linewidth=0.3, linestyle='-')
ax.tick_params(labelsize=10)

plt.title('Temperatura Media - 1983-2020', fontsize=12)

plt.tight_layout()
plt.show()

# ---------------------------------------------------------------------------- #
# ---------------------------------------------------------------------------- #