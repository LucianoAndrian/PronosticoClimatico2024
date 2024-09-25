"""Funciones para setear archivos de modelos"""
# import --------------------------------------------------------------------- #
import xarray as xr
import os
import glob
import re
import numpy as np
import pandas as pd
# ---------------------------------------------------------------------------- #
def SelectNMMEFiles(modelname, variable, dir, members=10):

    """Selecciona archivos de los modelos

    Parametros:
    modelname (str): nombre del modelo
    variable (str): nombre de la variable
    dir (str): directorio de los archivos
    members (int): cantidad de miembros de ensamble
    return (list): lista con todos los nombres de los archios seleccionados

    -----------------------
    Ejemplo de uso:
        SelectNMMEFiles(modelname='NCEP-CFSv2',
                        variable='prec',
                        dir=dir,
                        members=21)
    """

    files = glob.glob(f"{dir}{variable}_Amon_*{modelname}*_*_r*_*-*.nc")

    if members <= 9:
        pattern = r'_r[1-{}]_'.format(members)
    elif members <= 19:
        pattern = r'_r([1-9]|1[0-{}])_'.format(members - 10)
    else:
        pattern = r'_r([1-9]|1[0-9]|2[0-{}])_'.format(members - 20)

    files = [f for f in files if re.search(pattern, f)]

    files = sorted(files, key=lambda x: x.split()[0])

    if len(files) == 0:
        print('No se puedieron seleccionar archivos')

    return files

def fix_calendar(ds, timevar='time'):
    """Agrega los dias a los archivos de NMME

    Parametros:
    ds (xarray.Dataset): simulaciones del modelo
    timevar (str): nombre de la variable tiempo
    return (xarray.Dataset): simulaciones del modelo con fecha corregida
    """

    if ds[timevar].attrs['calendar'] == '360':
        ds[timevar].attrs['calendar'] = '360_day'
    return ds


def SelectSeason(files, ic_month):
    """Selecciona los archivos segun el mes de inicio

    Parametros:
    files (list): lista de caracteres
    ic_month (str): numero del mes de inicio del pronostico

    return files_sesaons (list): lista con los pronos seleccionados
    """
    pattern = f"{ic_month}_"
    files_seasons = [prono for prono in files if pattern in prono]
    return files_seasons


def SetModel(files, lons, lats, max_lead, forecast_lead=None, season=False,
             single_lead=None,):
    """Abre y setea los pronsticos de los modelos

    Parmetros:
    files (list): lista con nombres de archivos de modelos del NMME
    lons (list): lista con limites de longitud para seleccionar dominio
    lats (list): lista con limites de latitud
    max_lead (int): maximo lead considerado
    forecast_lead (int): Default None. Si es int selecciona a partir de ahi
    sigle_lead (int): Default None. Si es int se selecciona solo ese leadtime
    season (bool): Default False. Si es True promedia todos los leads
    centrando en el medio

    return (xarray.Dataset): pronosticos del modelo seteados

    ----------------------------------------------------------------------------
    Ejemplo de uso:
        SetModel(files=files,
                lons=[275, 330],
                lats=[-60, 15],
                max_lead=4,
                forecast_lead=1,
                seasons = True
                single_lead=None)
    """

    data = (xr.open_mfdataset(files, decode_times=False)
            .sel(X=slice(lons[0], lons[1]), Y=slice(lats[0], lats[1])))

    if len(data.Y) == 0:
        data = (xr.open_mfdataset(files, decode_times=False)
                .sel(X=slice(lons[0], lons[1]), Y=slice(lats[1], lats[0])))

    data = data.rename({'X': 'lon', 'Y': 'lat', 'M': 'r', 'S': 'time'})
    if single_lead is not None:
        data = data.sel(L=single_lead)
    else:
        max_lead += 1.5
        data = data.sel(L=np.arange(0.5, max_lead))

    data = xr.decode_cf(fix_calendar(data))

    if forecast_lead is not None:
        forecast_lead += 0.5
        data = data.sel(L=slice(forecast_lead, max_lead))

    if season is True:
        data = data.mean('L')

    return data
# ---------------------------------------------------------------------------- #

def SetModelAttrs(ds, variable_name, variable_standard_name,
                  variable_long_name, variable_units):
    """Setea los atributos del archivo .nc

    Parmetros:
    ds (xarray.Dataset): salida de SetModel
    variable_name (str): nombre de la variable
    variable_standard_name (str): standard_name de la variable
    variable_long_name (str): long_name de la variable
    variable_units (str): units de la variable

    return (xarray.Dataset): idem ds pero con atributos
    """

    if variable_name == 'prec':
        variable_name_grib = 'PRATE'
    elif variable_name == 'tref':
        variable_name_grib = 'TMP'

    ds['lon'].attrs = {
        "standard_name": "longitude",
        "pointwidth": 1.0,
        "long_name": "longitude",
        "gridtype": 1,
        "units": "degree_east"
    }

    ds['lat'].attrs = {
        "standard_name": "latitude",
        "pointwidth": 1.0,
        "long_name": "latitude",
        "gridtype": 0,
        "units": "degree_north"
    }

    # ds['time'].attrs = {
    #     "standard_name": "forecast_reference_time",
    #     "pointwidth": 0,
    #     "long_name": "Forecast Start Time",
    #     "calendar": "360",
    #     "gridtype": 0,
    #     "units": "months since 1960-01-01"
    # }

    ds['r'].attrs = {
        "standard_name": "realization",
        "pointwidth": 1.0,
        "long_name": "Ensemble Member",
        "gridtype": 0,
        "units": "unitless"
    }

    ds[variable_name].attrs = {
        "pointwidth": 0,
        "grib_name": variable_name_grib,
        "gribPDSpattern": "04XXXX",
        "standard_name": variable_standard_name,
        "long_name": variable_long_name,
        "file_missing_value": 9.999e+20,
        "missing_value": float('nan'),
        "units": variable_units
    }

    return ds


def SetIndex(file_path, variable_name, skipfooter=5):
    col_names = ['Year'] + [str(i).zfill(2) for i in range(1, 13)]

    df_main = pd.read_csv(
        file_path,
        sep="\\s+",
        skiprows=1,
        skipfooter=skipfooter,
        names=col_names, usecols=range(13), engine='python'
    )

    # Reordenando
    df_long = df_main.melt(id_vars='Year', var_name='month',
                           value_name=variable_name)

    df_long['time'] = pd.to_datetime(
        df_long['Year'].astype(str) +
        df_long['month'].astype(str).str.zfill(2), format='%Y%m')

    df_long = df_long.drop(columns=['Year', 'month'])
    df_long = df_long.set_index(['time']).sort_index()
    ds = df_long.to_xarray()

    # Info extra
    with open(file_path, 'r') as file:
        lines = file.readlines()

    extra_lines = [line.strip() for line in lines[-skipfooter:]]
    df_extra = pd.DataFrame(extra_lines, columns=['Content'])

    return ds, df_extra


def wget_fun(url, namefile, out_dir):
    """Descarga con wget

    Parametros:
    url (str): url
    namefile (str): nombre con el que se va guardar el archiv
    out_dir (str): directorio donde se va guardar
    """

    try:
        os.system(f'wget -O {out_dir}{namefile} {url}')
    except:
        print(f"Error al intentar descargar {out_dir}{namefile} {url}")

