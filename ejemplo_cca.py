"""
Ejemplo de uso de funciones de CCA.



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

# SST - ASO
ds1 = xr.open_dataset(f'{ruta}sst_ERSSTv5_1980-2020.nc')
ds1 = ds1.rolling(time=3, center=True).mean('time')
ds1 = ds1.sel(time=ds1.time.dt.year.isin(np.arange(1984,2020)))
ds1 = ds1.sel(lon=slice(150, 280), lat=slice(15,-15))
sst_or = ds1.sel(time=ds1.time.dt.month.isin(9))

# SLP - ASO
ds3 = xr.open_dataset(f'{ruta}slp_NCEP_1980-2020.nc')
ds3 = ds3.rolling(time=3, center=True).mean('time')
ds3 = ds3.sel(time=ds3.time.dt.year.isin(np.arange(1984,2020)))
ds3 = ds3.sel(lon=slice(210,320), lat=slice(-20,-75))
slp_or = ds3.sel(time=ds3.time.dt.month.isin(9))

# PP - OND
ds2 = xr.open_dataset(f'{ruta}prec_monthly_nmme_cpc_sa.nc')
ds2 = ds2.rolling(time=3, center=True).mean('time')
ds2 = ds2.sel(time=ds2.time.dt.year.isin(np.arange(1984,2020)))
pp_or = ds2.sel(time=ds2.time.dt.month.isin(11))

# Anomalia cv y normalizo
sst = CrossAnomaly_1y(sst_or, norm=True).sst
pp = CrossAnomaly_1y(pp_or, norm=False).prec
slp = CrossAnomaly_1y(slp_or, norm=False).slp

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
plt.plot(A[:,0], label='Vector Canonico de SLP')
plt.plot(B[:,0], label='Vector Canonico de PP')
plt.legend()
plt.title(f'Modo 0 - r={S[0]:.3g}')
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

pp_training = pp_or.sel(time=pp_or.time.dt.year.isin(np.arange(1983,2011)))
pp_testing = pp_or.sel(time=pp_or.time.dt.year.isin(np.arange(2011,2020)))
slp_training = slp_or.sel(time=slp_or.time.dt.year.isin(np.arange(1983,2011)))
slp_testing = slp_or.sel(time=slp_or.time.dt.year.isin(np.arange(2011,2020)))

sst_training = sst_or.sel(time=sst_or.time.dt.year.isin(np.arange(1983,2011)))
sst_testing = sst_or.sel(time=sst_or.time.dt.year.isin(np.arange(2011,2020)))
#(probar como da reemplazando slp por sst)

var_exp = 0.7 # varianza que queremos retener
adj, b_verif = CCA_mod(X=slp_training, X_test=slp_testing,
                       Y=pp_training, var_exp=var_exp)

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

adj_escalado = adj_xr.sum('modo')*pp_training.std('time').prec/(var_exp**2)

# Reconstruido
Plot(adj_xr, adj_escalado.sel(time='2015-11-01')[0,:,:],
     np.arange(-120, 120, 20), cmap='BrBG',
     title='PP - Reconstruido: OND 2015')

# Observado
observado = pp_testing-pp_training.mean('time')
Plot(pp_testing, observado.sel(time='2015-11-01').prec[0,:,:],
     np.arange(-100, 120, 20), cmap='BrBG',
     title='PP - Observado: OND 2015')


adj_escalado = adj_xr.sum('modo')*pp_training.std('time').prec

# Reconstruido SIN ESCALAR POR LA VARIANZA
Plot(adj_xr, adj_escalado.sel(time='2015-11-01')[0,:,:],
     np.arange(-120, 120, 20), cmap='BrBG',
     title='PP - Reconstruido sin escalar por la varianza: OND 2015')
# ---------------------------------------------------------------------------- #
# ---------------------------------------------------------------------------- #
# Calibración de Pronostico
# ---------------------------------------------------------------------------- #
ruta_mod = '~/PronoClim/modelos_seteados/'
mod_gem = xr.open_dataset(ruta_mod + 'prec_CMC-CanCM4i-IC3_SON.nc')*(92/3)
mod_gem = mod_gem.sel(time=mod_gem.time.dt.year.isin(np.arange(1984,2020)))
mod_gem_em = mod_gem.mean('r') # media del ensamble

pp_or_aux = ds2.sel(time=ds2.time.dt.month.isin(10)) # pp observada en DJF

# nuevamente separamos en training y testing
pp_training = pp_or_aux.sel(
    time=pp_or_aux.time.dt.year.isin(np.arange(1984,2011)))
pp_testing = pp_or_aux.sel(
    time=pp_or_aux.time.dt.year.isin(np.arange(2011,2020)))
mod_gem_em_training = mod_gem_em.sel(
    time=mod_gem_em.time.dt.year.isin(np.arange(1984,2011)))
mod_gem_testing = mod_gem.sel(
    time=mod_gem.time.dt.year.isin(np.arange(2011,2020)))


var_exp = 0.7 # varianza que queremos retener
pp_mod_adj = []
for r in mod_gem.r.values:
    adj, b_verif = CCA_mod(X=mod_gem_em_training,
                           X_test=mod_gem_testing.sel(r=r),
                           Y=pp_training, var_exp=var_exp)

    adj_rs = adj.reshape(adj.shape[0],
                         len(pp.lat.values),
                         len(pp.lon.values),
                         adj.shape[2])

    adj_xr = xr.DataArray(adj_rs, dims=['time', 'lat', 'lon', 'modo'],
                          coords={'lat': pp.lat.values,
                                  'lon': pp.lon.values,
                                  'time': mod_gem_testing.time.values,
                                  'modo': np.arange(0, adj_rs.shape[-1])})

    # sumamos todos los modos y escalamos para reconstruir los datos
    adj_xr = adj_xr.sum('modo') * mod_gem_em_training.std('time').prec/(var_exp**2)

    pp_mod_adj.append(adj_xr)

# Concatenamos los valores, le damos valores a la variable "r" y
# y pasamos a xr.Dataset para tener una variable del mismo tipo que la original
pp_mod_adj = xr.concat(pp_mod_adj, dim='r')
pp_mod_adj['r'] = mod_gem_testing['r']
pp_mod_gem_adj = pp_mod_adj.to_dataset(name='prec')


Plot(pp_mod_gem_adj,
     mod_gem.sel(time='2015-08-01').mean('r').prec[0,:,:]-mod_gem_em.mean('time').prec,
     np.arange(-60, 65, 5), cmap='BrBG',
     title='PP - Pronostico SON 2015')

Plot(pp_mod_gem_adj, pp_mod_gem_adj.sel(time='2015-08-01').mean('r').prec[0,:,:],
     np.arange(-60, 65, 5), cmap='BrBG',
     title='PP - Pronostico Calibrado SON 2015')

Plot(pp_testing,
     pp_testing.sel(time='2015-10-01').prec[0,:,:]-pp_training.mean('time').prec,
     np.arange(-60, 65, 5), cmap='BrBG',
     title='PP - Observado SON 2015')























