"""
Estimación de la temperatura y precipitacion media mensual a partir de datos
de estaciones del SMN.
Los datos originales estan en temp. max y min de cada día y pp de cada dia.
(Decepción, falencias y frustración.)
"""
# ---------------------------------------------------------------------------- #
import pandas as pd
import xarray as xr
# ---------------------------------------------------------------------------- #
# Leer el archivo CSV (ajusta la ruta al archivo que tengas)
archivo = 'datos_climaticos.csv'

# Cargar los datos en un DataFrame
df = pd.read_csv('~/PronoClim/datos_estaciones/Exp.Nro.205222-1.txt')
df2 = pd.read_csv('~/PronoClim/datos_estaciones/Exp.Nro.205222.txt')

# ---------------------------------------------------------------------------- #
df['Fecha'] = pd.to_datetime(df['Fecha'])
df2['Fecha'] = pd.to_datetime(df2['Fecha'])

df['Temp. Maxima (°C)'] = (
    pd.to_numeric(df['Temp. Maxima (°C)'], errors='coerce'))
df['Temp. Minima (°C)'] = (
    pd.to_numeric(df['Temp. Minima (°C)'], errors='coerce'))
df['Precipitacion (mm)'] = (
    pd.to_numeric(df['Precipitacion (mm)'], errors='coerce'))

df2['Temp. Maxima (°C)'] = (
    pd.to_numeric(df2['Temp. Maxima (°C)'], errors='coerce'))
df2['Temp. Minima (°C)'] = (
    pd.to_numeric(df2['Temp. Minima (°C)'], errors='coerce'))
df2['Precipitacion (mm)'] = (
    pd.to_numeric(df2['Precipitacion (mm)'], errors='coerce'))

df_concat = pd.concat([df, df2], ignore_index=True)
df_concat_ordenado = df_concat.sort_values(by=['Estacion', 'Fecha'])

# Temp "media" diaria
df_concat_ordenado['Temp. Media (°C)'] = (
        (df_concat_ordenado['Temp. Maxima (°C)'] +
         df_concat_ordenado['Temp. Minima (°C)']) / 2)

# Agrupando por año y mes
df_concat_ordenado['Anio-Mes'] = df_concat_ordenado['Fecha'].dt.to_period('M')

# Agrupando prr Estacion y Anio-Mes para la media mensual de T
promedio_temp_media = df_concat_ordenado.groupby(
    ['Estacion', 'Anio-Mes'])['Temp. Media (°C)'].mean().reset_index()

# Media de PP sin los ceros
df_precip_sin_ceros = df_concat_ordenado[
    df_concat_ordenado['Precipitacion (mm)'] > 0]
# Agrupando prr Estacion y Anio-Mes para la media mensual de PP
promedio_precipitacion = df_precip_sin_ceros.groupby(
    ['Estacion', 'Anio-Mes'])['Precipitacion (mm)'].mean().reset_index()


promedios_mensuales = pd.merge(promedio_temp_media, promedio_precipitacion,
                               on=['Estacion', 'Anio-Mes'], how='left')
promedios_mensuales.rename(
    columns={'Temp. Media (°C)': 'Temp. Media Mensual (°C)',
             'Precipitacion (mm)': 'Precipitacion Media Mensual (mm)'},
    inplace=True)

# xr.Dataset ----------------------------------------------------------------- #
promedios_mensuales['time'] = pd.to_datetime(
    promedios_mensuales['Anio-Mes'].astype(str) + '-01')

ds = xr.Dataset(
    data_vars={
        'temp': (('estacion', 'time'), promedios_mensuales.pivot(
            index='Estacion', columns='time',
            values='Temp. Media Mensual (°C)').values),
        'prec': (('estacion', 'time'), promedios_mensuales.pivot(
            index='Estacion', columns='time',
            values='Precipitacion Media Mensual (mm)').values)
    },
    coords={
        'estacion': promedios_mensuales['Estacion'].unique(),
        'time': pd.to_datetime(promedios_mensuales['time'].unique())
    }
)

ds['estacion'].attrs = {
    "standard_name": 'estacion'
}

ds['temp'].attrs = {
    "standard_name": 'temp',
    "long_name": "temperatura media mensual",
    "units" : "ºC",
    "missing_value": "nan"
}

ds['prec'].attrs = {
    "standard_name": 'prec',
    "long_name": "precipitacion media mensual",
    "units" : "mm",
    "missing_value": "nan",
    "data": "No se cuentan los dias sin lluvia",
}

ds.to_netcdf('/home/luciano.andrian/PronoClim/obs_seteadas/datos_estaciones_SMN.nc')
# ---------------------------------------------------------------------------- #