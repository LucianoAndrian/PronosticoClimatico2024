"""Descarga y setea indices"""
# ---------------------------------------------------------------------------- #
from funciones import SetIndex, wget_fun

# Seteos generales ----------------------------------------------------------- #
save=True

filedir = "/home/luciano.andrian/PronoClim/indices_descargados/"
outdir =  "/home/luciano.andrian/PronoClim/indices_nc/"
################################################################################

# Nino 3.4 ------------------------------------------------------------------- #
wget_fun(url='https://psl.noaa.gov/gcos_wgsp/Timeseries/Data/nino34.long.data',
         namefile='nino34.txt', out_dir=filedir)

ds_n34, df_extra = SetIndex(filedir + 'nino34.txt',
                                variable_name='nino34')

ds_n34['nino34'].attrs = {
    "standard_name": 'nino34',
    "long_name": 'Monthly SST nino34 region',
    "file_missing_value": df_extra['Content'][0],
    "missing_value": df_extra['Content'][0],
    "units": 'ºC',
    "source_domain": df_extra['Content'][2],
    "source_dataset": df_extra['Content'][3],
    "url": df_extra['Content'][4]
}

ds_n34 = ds_n34.sel(time=slice(None, '2020-12-01'))
ds_n34.to_netcdf(f"{outdir}nino34_mmean.nc")

# Nino 3 --------------------------------------------------------------------- #
wget_fun(url='https://psl.noaa.gov/gcos_wgsp/Timeseries/Data/nino3.long.data',
         namefile='nino3.txt', out_dir=filedir)

ds_n3, df_extra = SetIndex(filedir + 'nino3.txt', variable_name='nino3')

ds_n3['nino3'].attrs = {
    "standard_name": 'nino3',
    "long_name": 'Monthly SST nino3 region',
    "file_missing_value": df_extra['Content'][0],
    "missing_value": df_extra['Content'][0],
    "units": 'ºC',
    "source_domain": df_extra['Content'][2],
    "source_dataset": df_extra['Content'][3],
    "url": df_extra['Content'][4]
}
ds_n3 = ds_n3.sel(time=slice(None, '2020-12-01'))
ds_n3.to_netcdf(f"{outdir}nino3_mmean.nc")

# SOI ------------------------------------------------------------------------ #
wget_fun(url='https://psl.noaa.gov/gcos_wgsp/Timeseries/Data/soi.long.data',
         namefile='soi.txt', out_dir=filedir)

ds_soi, df_extra = SetIndex(filedir + 'soi.txt', variable_name='soi',
                            skipfooter=9)

ds_soi['soi'].attrs = {
    "standard_name": 'SOI',
    "long_name": 'Southern Oscillation Index',
    "file_missing_value": df_extra['Content'][0],
    "missing_value": df_extra['Content'][0],
    "units": 'norm',
    "source_dataset": f"{df_extra['Content'][1]}: {df_extra['Content'][5]}",
    "url": df_extra['Content'][3],
    "comment": f"{df_extra['Content'][6]} {df_extra['Content'][7]}"
}

ds_soi = ds_soi.sel(time=slice(None, '2020-12-01'))
ds_soi.to_netcdf(f"{outdir}soi_mmean.nc")

# IOD ------------------------------------------------------------------------ #
wget_fun(url='https://psl.noaa.gov/gcos_wgsp/Timeseries/Data/dmi.had.long.data',
         namefile='dmi.txt', out_dir=filedir)

ds_iod, df_extra = SetIndex(filedir + 'dmi.txt', variable_name='dmi',
                            skipfooter=7)

ds_iod['dmi'].attrs = {
    "standard_name": 'dmi',
    "long_name": 'Dipole Mode Index',
    "file_missing_value": df_extra['Content'][0],
    "missing_value": df_extra['Content'][0],
    "climatology":  "Climatology is currently 1981-2010",
    "units": 'ºC',
    "source_dataset": f"{df_extra['Content'][1]}",
    "url": df_extra['Content'][5],
    "comment": f"DMI {df_extra['Content'][3]}"
}

ds_iod = ds_iod.sel(time=slice(None, '2020-12-01'))
ds_iod.to_netcdf(f"{outdir}iod_mmean.nc")

# SAM ------------------------------------------------------------------------ #
wget_fun(url='http://www.nerc-bas.ac.uk/public/icd/gjma/newsam.1957.2007.txt',
         namefile='sam.txt', out_dir=filedir)

ds_sam, df_extra = SetIndex(filedir + 'sam.txt', variable_name='sam',
                            skipfooter=0)

ds_sam['sam'].attrs = {
    "standard_name": 'sam',
    "long_name": 'Southern Annular Mode',
    "file_missing_value": 'nan',
    "missing_value": 'monthly mean for the 1971-2000 period',
    "units": 'hPa',
    "url": 'https://legacy.bas.ac.uk/met/gjma/sam.html'
}

ds_sam = ds_sam.sel(time=slice(None, '2020-12-01'))
ds_sam.to_netcdf(f"{outdir}sam_mmean.nc")

# ---------------------------------------------------------------------------- #
# ---------------------------------------------------------------------------- #