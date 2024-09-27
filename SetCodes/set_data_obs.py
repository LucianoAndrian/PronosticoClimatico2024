"""Seteos de bases de datos"""
# ---------------------------------------------------------------------------- #
import xarray as xr
import cftime

# ---------------------------------------------------------------------------- #
def change_day_to_first(date):
    return cftime.Datetime360Day(date.year, date.month, 1)

# Seteos generales ----------------------------------------------------------- #
save = True

# Dominios y leads:
lons = [275, 330]
lats = [-60, 20]

# ---------------------------------------------------------------------------- #
dir = '/pikachu/datos/osman/ereg/descargas/NMME/'
out_dir = '/home/luciano.andrian/PronoClim/obs_seteadas/'

# ---------------------------------------------------------------------------- #
# Tref
data = xr.open_dataset(dir + 'tref_monthly_nmme_ghcn_cams.nc')
data = data.rename(T='time', X='lon', Y='lat')
data['time'] = [change_day_to_first(t) for t in data['time'].values]
data = data.sel(time=slice('1983-01-01', '2020-12-01'),
                lon=slice(*lons), lat=slice(*lats))
if save:
    data.to_netcdf(out_dir + 'tref_monthly_nmme_ghcn_cams_sa.nc')

# Prec
data = xr.open_dataset(dir + 'prec_monthly_nmme_cpc.nc')
data = data.rename(T='time', X='lon', Y='lat')
data['time'] = [change_day_to_first(t) for t in data['time'].values]
data = data.sel(time=slice('1983-01-01', '2020-12-01'),
                lon=slice(*lons), lat=slice(*lats))

if save:
    data.to_netcdf(out_dir + 'prec_monthly_nmme_cpc_sa.nc')
# ---------------------------------------------------------------------------- #