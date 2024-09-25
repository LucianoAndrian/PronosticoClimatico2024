"""Seteos de modelos del NMME"""
# ---------------------------------------------------------------------------- #
from funciones import SelectNMMEFiles, SetModel, SetModelAttrs, SelectSeason

# Seteos generales ----------------------------------------------------------- #
save = True

# Dominios y leads:
lons = [275, 330]
lats = [-60, 20]
lead_maximo = 3 # ej. Agosto --> ago (0), sep(1), oct (2), nov (3)
miembros_de_ensamble = 20

ic_months = ['02','05','08','11']
seasons_name = ['MAM', 'JJA', 'SON', 'DJF']

# ---------------------------------------------------------------------------- #
dir_hc = '/pikachu/datos/osman/nmme/monthly/hindcast/'
out_dir = '/home/luciano.andrian/PronoClim/modelos_seteados/'

# GEM5-NEMO ####################################################################
# Precipitacion -------------------------------------------------------------- #

for name_model in ['GEM5-NEMO', 'CanCM4i-IC3']:
    for v_name in ['prec', 'tref']:
        files = SelectNMMEFiles(modelname=name_model,
                                variable=v_name, dir=dir_hc,
                                members=miembros_de_ensamble)

        for m, n_s in zip(ic_months, seasons_name):
            files_seasons = SelectSeason(files, m)

            data = SetModel(files_seasons, lons, lats, lead_maximo,
                            forecast_lead=1,
                            season=True,
                            single_lead=None)

            data = data.sel(time=slice('1983-01-01', '2020-12-01'))

            data['r'] = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]

            if v_name == 'prec':
                data = SetModelAttrs(
                    data, variable_name='prec',
                    variable_standard_name='lwe_precipitation_rate',
                    variable_long_name='Total Precipitation',
                    variable_units='mm/day')
            else:
                data = SetModelAttrs(
                    data, variable_name='tref',
                    variable_standard_name='air_temperature',
                    variable_long_name='Reference Temperature',
                    variable_units='Kelvin_scale')

            if save:
                data.to_netcdf(f"{out_dir}{v_name}_CMC-{name_model}_{n_s}.nc")

# ---------------------------------------------------------------------------- #