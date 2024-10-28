"""
Ejemplo de uso de funciones de CCA.

AVISO: Las funciones van a tener algunos upgrades para el uso de ensambles.
Esto aún no está disponible.
"""
# ---------------------------------------------------------------------------- #
import numpy as np
import xarray as xr
from funciones_practicas import CrossAnomaly_1y, CCA, CCA_mod

# Funcion auxiliar para plotear ---------------------------------------------- #
# (adaptada de PlotContourf_SA)
def Plot(data, data_var, scale, cmap, title = ''):

    import matplotlib.pyplot as plt
    import numpy as np
    from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
    import cartopy.crs as ccrs

    crs_latlon = ccrs.PlateCarree()

    ratio = len(data.lat)/len(data.lon)

    fig = plt.figure(figsize=(6, 6*ratio), dpi=100)

    ax = plt.axes(projection=ccrs.PlateCarree(central_longitude=180))
    ax.set_extent([data.lon.values[0],
                   data.lon.values[-1],
                   min(data.lat.values),
                   max(data.lat.values)], crs=crs_latlon)

    # Contornos
    im = ax.contourf(data.lon,
                     data.lat,
                     data_var,
                     levels=scale,
                     transform=crs_latlon, cmap=cmap, extend='both')

    # barra de colores
    cb = plt.colorbar(im, fraction=0.042, pad=0.035, shrink=0.8)
    cb.ax.tick_params(labelsize=8)

    ax.coastlines(color='k', linestyle='-', alpha=1)

    ax.set_xticks(np.arange(data.lon.values[0], data.lon.values[-1], 20),
                  crs=crs_latlon)
    ax.set_yticks(np.arange(min(data.lat.values), max(data.lat.values), 10),
                  crs=crs_latlon)
    lon_formatter = LongitudeFormatter(zero_direction_label=True)
    lat_formatter = LatitudeFormatter()
    ax.xaxis.set_major_formatter(lon_formatter)
    ax.yaxis.set_major_formatter(lat_formatter)

    ax.gridlines(crs=crs_latlon, linewidth=0.3, linestyle='-')
    ax.tick_params(labelsize=10)

    plt.title(title, fontsize=12)

    plt.tight_layout()
    plt.show()
# ---------------------------------------------------------------------------- #
ruta = '~/PronoClim/obs_seteadas/'

# SST - SON
ds1 = xr.open_dataset(f'{ruta}sst_ERSSTv5_1980-2020.nc')
ds1 = ds1.rolling(time=3, center=True).mean('time')
ds1 = ds1.sel(time=ds1.time.dt.year.isin(np.arange(1983,2020)))
ds1 = ds1.sel(lon=slice(150, 280), lat=slice(15,-15))
ds1 = ds1.sel(time=ds1.time.dt.month.isin(10))

# SLP - SON
ds3 = xr.open_dataset(f'{ruta}slp_NCEP_1980-2020.nc')
ds3 = ds3.rolling(time=3, center=True).mean('time')
ds3 = ds3.sel(time=ds3.time.dt.year.isin(np.arange(1983,2020)))
ds3 = ds3.sel(time=ds3.time.dt.month.isin(10))
ds3 = ds3.sel(lon=slice(210,320), lat=slice(-20,-75))

# PP - OND
ds2 = xr.open_dataset(f'{ruta}prec_monthly_nmme_cpc_sa.nc')
ds2 = ds2.rolling(time=3, center=True).mean('time')
ds2 = ds2.sel(time=ds2.time.dt.year.isin(np.arange(1983,2020)))
ds2 = ds2.sel(time=ds2.time.dt.month.isin(11))

# Anomalia cv y normalizo
sst = CrossAnomaly_1y(ds1, norm=True).sst
pp = CrossAnomaly_1y(ds2, norm=True).prec
slp = CrossAnomaly_1y(ds3, norm=True).slp

# ---------------------------------------------------------------------------- #
# Uso de CCA ----------------------------------------------------------------- #
# var_exp=0.7 va seleccionar las componentes principales necesarias para
# explicar el 70% de la varianza. Eso lo hace de forma separada para X e Y
P, Q, heP, heQ, S, A, B = CCA(X=slp, Y=pp, var_exp=0.7)

# Obtengo np.ndarrays:
# P y Q mapas canonicos de X e Y, respectivamente
# heP, heQ: varianzas heterogeneas
# S: valores singulares/correlacion entre A y B
# A y B: vectores canonicos de X e Y, respectivamente

# Los mapas canonicos tienen dimencion (t, n) donde n=lon*lat
# Depende de que se quiera hacer con ellos luego podemos necesitar recuperarlos
# en las dimensiones (t, lon, lat)

print(P.shape)
# Hay 4 vectores canonicos, indicando que el metodo selecciono 2 componentes
# principales

# Para P, que es el mapa canonico de sst
aux = P.reshape(len(slp.lat.values), len(slp.lon.values), P.shape[1])
P_da = xr.DataArray(aux, dims=['lat', 'lon', 'cca'],
                    coords={'lat': slp.lat.values,
                            'lon': slp.lon.values})

Plot(P_da, P_da.sel(cca=0), np.arange(-1, 1.2, .2), cmap='RdBu_r',
     title='SLP Modo=0')

Plot(P_da, P_da.sel(cca=1), np.arange(-1, 1.2, .2), cmap='RdBu_r',
     title='SLP Modo=1')

# idem con Q
aux = Q.reshape(len(pp.lat.values), len(pp.lon.values), Q.shape[1])
Q_da = xr.DataArray(aux, dims=['lat', 'lon', 'cca'],
                    coords={'lat': pp.lat.values,
                            'lon': pp.lon.values})
Plot(Q_da, Q_da.sel(cca=0), np.arange(-1, 1.2, .2), cmap='BrBG',
     title='PP Modo=0')

Plot(Q_da, Q_da.sel(cca=1), np.arange(-1, 1.2, .2), cmap='BrBG',
     title='PP Modo=1')

# Podemos ver la evolución temporal de estos campos en el tiempo
import matplotlib.pyplot as plt
plt.plot(A[:,0], label='Vector Canonico de SST')
plt.plot(B[:,0], label='Vector Canonico de PP')
plt.legend()
plt.title(f'Modo 0 - r={np.round(S[0],3)}')
plt.xlabel('tiempo')
plt.show()
# S indica la correlacion entre los vectores canonicos
# podemos verlo:
print(f"Correlación A y B modo 0: {np.corrcoef(A[:,0], B[:,0])[0,1]}")
print(f"Valor de S[0] = {S[0]}")

# ---------------------------------------------------------------------------- #
# Probemos con sst y pp
P, Q, heP, heQ, S, A, B = CCA(X=sst, Y=pp, var_exp=0.7)
aux = P.reshape(len(sst.lat.values), len(sst.lon.values), P.shape[1])
P_da = xr.DataArray(aux, dims=['lat', 'lon', 'cca'],
                    coords={'lat': sst.lat.values,
                            'lon': sst.lon.values})

Plot(P_da.sel(cca=0), P_da.sel(cca=0), np.arange(-1, 1.2, .2), cmap='RdBu_r')

aux = Q.reshape(len(pp.lat.values), len(pp.lon.values), Q.shape[1])
Q_da = xr.DataArray(aux, dims=['lat', 'lon', 'cca'],
                    coords={'lat': pp.lat.values,
                            'lon': pp.lon.values})
Plot(Q_da.sel(cca=0), Q_da.sel(cca=0), np.arange(-1, 1.2, .2), cmap='BrBG')


# ---------------------------------------------------------------------------- #
# Uso de CCA_mod ------------------------------------------------------------- #
# Similar a CCA(), pero permite reconstruir/pronosticar a partir de un predictor
# por medio de CCA.

# Creamos training y testing (para reconstruir en ese periodo)

pp_sel = pp.sel(time=pp.time.dt.year.isin(np.arange(1983,2011)))
pp_testing = pp.sel(time=pp.time.dt.year.isin(np.arange(2011,2021)))
slp_sel = slp.sel(time=slp.time.dt.year.isin(np.arange(1983,2011)))
slp_testing = slp.sel(time=slp.time.dt.year.isin(np.arange(2011,2021)))

sst_sel = sst.sel(time=sst.time.dt.year.isin(np.arange(1983,2011)))
sst_testing = sst.sel(time=sst.time.dt.year.isin(np.arange(2011,2021)))

# Probar reemplazando slp por sst
adj, b_verif = CCA_mod(X=slp_sel, X_test=slp_testing, Y=pp_sel, var_exp=0.7)
# Obtenemos
# adj np.ndarray de dimensiones (tiempo_testing, lon_pp*lat_pp, modos)
# evolucion temporal de los modos reconstruidos
print(adj.shape)
# la organizacion de adj puede ser modificada a gusto en la ultima parte de la
# funcion CCA_mod para los propositos que se necesiten

# Reorganizo adj para tener (t, lon, lat, modos)

adj_rs = adj.reshape(adj.shape[0],
                     len(pp.lat.values),
                     len(pp.lon.values),
                     adj.shape[2])

adj_xr = xr.DataArray(adj_rs, dims=['time','lat', 'lon', 'modo'],
                          coords={'lat': pp.lat.values,
                                  'lon': pp.lon.values,
                                  'time': pp_testing.time.values,
                                  'modo':np.arange(0, adj_rs.shape[-1])})
print(adj_xr.dims)

# Reconstruido
Plot(adj_xr, adj_xr.sel(time='2015-11-01', modo=0)[0,:,:],
     np.arange(-1, 1.2, 0.2), cmap='BrBG', title='PP - Reconstruido: OND 2015')

# Observado
Plot(pp_testing, pp_testing.sel(time='2015-11-01')[0,:,:],
     np.arange(-1, 1.2, 0.2)*2, cmap='BrBG', title='PP - Observado: OND 2015')
# ---------------------------------------------------------------------------- #
# ---------------------------------------------------------------------------- #