#!/usr/bin/env python

import configparser
import oracledb
import os
import pandas as pd
import duckdb
import time
import warnings
warnings.filterwarnings("ignore", category=UserWarning, message=".*pandas only supports SQLAlchemy.*")


def elapsed_time(t1, verb=False):
    """
    Returns the time between t1 and the current time now
    I can can also print the formatted elapsed time.
    ----------
    t1: float
        The initial time (in seconds)
    verb: bool, optional
        Optionally print the formatted elapsed time
    returns
    -------
    stime: float
        The elapsed time in seconds since t1
    """
    t2 = time.time()
    stime = "%dm %2.2fs" % (int((t2-t1)/60.), (t2-t1) - 60*int((t2-t1)/60.))
    if verb:
        print("Elapsed time: {}".format(stime))
    return stime


def load_db_config(config_file, profile):
    config = configparser.ConfigParser()
    config.read(config_file)

    section = dict(config[profile])
    section['dsn'] = f'{section["server"]}:{section["port"]}/{section["name"]}'
    return section


db_section = 'db-decade'
schema = 'desoper'
# db_section = 'db-desoper'
# schema = 'prod'
config_file = os.path.join(os.environ['HOME'], 'dbconfig.ini')
# Get the connection credentials and information
creds = load_db_config(config_file, db_section)
dbh = oracledb.connect(user=creds['user'],
                       password=creds['passwd'],
                       dsn=creds['dsn'])


# Create the DR3_COADD_FILEPATH parquet table directly from a query
query = {}
query['DR3_COADDTILE_GEOM'] = "SELECT * FROM COADDTILE_GEOM"
query['DR3_COADD_IMAGE_FILEPATH'] = """
select c.FILENAME, c.TILENAME, c.BAND, c.FILETYPE, f.PATH, f.COMPRESSION,
       t.CROSSRA0, t.RACMIN, t.RACMAX, t.DECCMIN, t.DECCMAX,
       t.RA_CENT, t.DEC_CENT, t.RA_SIZE, t.DEC_SIZE,
       t.RAC1, t.RAC2, t.RAC3, t.RAC4, t.DECC1, t.DECC2, t.DECC3, t.DECC4
 from COADD c, PROCTAG p, FILE_ARCHIVE_INFO f, COADDTILE_GEOM t
  where p.TAG = 'DR3_COADD'
    and t.tilename = c.TILENAME
    and c.FILETYPE = 'coadd'
    and p.PFW_ATTEMPT_ID = c.PFW_ATTEMPT_ID
    and f.FILENAME = c.FILENAME"""

query["DR3_COADD_CATALOG_FILEPATH"] = """
select c.FILENAME, c.TILENAME, c.BAND, c.FILETYPE, f.PATH
  from CATALOG c, PROCTAG p, FILE_ARCHIVE_INFO f
   where p.TAG = 'DR3_COADD'
     and c.FILENAME = f.FILENAME
     and c.FILETYPE='coadd_cat'
     and p.PFW_ATTEMPT_ID = c.PFW_ATTEMPT_ID"""

query["DR3_COADD_TIFF_FILEPATH"] = """
select m.FILENAME, f.PATH, m.TILENAME
  from MISCFILE m, FILE_ARCHIVE_INFO f, PROCTAG p
   where p.TAG = 'DR3_COADD'
   and m.FILETYPE='coadd_tiff'
   and p.PFW_ATTEMPT_ID = m.PFW_ATTEMPT_ID
   and m.FILENAME = f.FILENAME"""

query["DR3_FINALCUT_IMAGE_FILEPATH"] = """
select i.FILENAME, f.PATH, f.COMPRESSION, i.BAND, i.EXPTIME, i.AIRMASS,
       i.FWHM, i.NITE, i.EXPNUM, i.CCDNUM,
       e.DATE_OBS, e.MJD_OBS,
       z.MAG_ZERO, z.SIGMA_MAG_ZERO,
       i.CROSSRA0, i.RACMIN, i.RACMAX, i.DECCMIN, i.DECCMAX,
       i.RA_CENT, i.DEC_CENT,
       i.RAC1, i.RAC2, i.RAC3, i.RAC4, i.DECC1, i.DECC2, i.DECC3, i.DECC4,
       (case when i.CROSSRA0='Y' THEN abs(i.RACMAX - (i.RACMIN-360)) ELSE abs(i.RACMAX - i.RACMIN) END) as RA_SIZE,
       abs(i.DECCMAX - i.DECCMIN) as DEC_SIZE
 from IMAGE i, EXPOSURE e, FILE_ARCHIVE_INFO f, PROCTAG p, ZEROPOINT z
  where p.TAG = 'DR3_FINALCUT'
   and p.pfw_attempt_id = i.pfw_attempt_id
   and i.EXPNUM=e.EXPNUM
   and i.FILENAME=f.FILENAME
   and i.FILETYPE='red_immask'
   and i.FILENAME=z.IMAGENAME
   """

query["DR3_FINALCUT_CATALOG_FILEPATH"] = """
select c.FILENAME, f.PATH, c.FILETYPE, c.BAND, c.CCDNUM
 from CATALOG c, FILE_ARCHIVE_INFO f, PROCTAG p
  where p.TAG = 'DR3_FINALCUT'
    and p.pfw_attempt_id = c.pfw_attempt_id
    and f.FILENAME=c.FILENAME
    and c.FILETYPE='cat_finalcut'"""


# The longer way, queries + parquet at the same time
tables = query.keys()
# tables = ['DR3_COADD_TIFF_FILEPATH']
# tables = ['DR3_FINALCUT_IMAGE_FILEPATH']
# tables = ['DR3_COADD_IMAGE_FILEPATH']
for parquet_name in tables:
    t0 = time.time()
    q = query[parquet_name]
    print(f"-- Will run query: {q}\n")
    df = pd.read_sql(q, dbh)
    print(f"-- Done with query in {elapsed_time(t0)}[s]")
    if 'PATH' in df.columns:
        df['PATH'] = df['PATH'].str.replace('DEC_Taiga/', 'DEC/', regex=False)
        df['PATH'] = df['PATH'].str.replace('ACT_Taiga/', 'ACT/', regex=False)
    df.to_parquet(f"{parquet_name}.parquet", engine="pyarrow", compression="snappy", index=True)

con = duckdb.connect("decade_metadata.duckdb")
for parquet_name in tables:
    t0 = time.time()
    query = f"CREATE OR REPLACE TABLE {parquet_name} AS SELECT * FROM '{parquet_name}.parquet'"
    con.execute(query)
    print(f"Wrote DuckDB table: {parquet_name} in {elapsed_time(t0)}[s]")

con.execute("VACUUM")  # Ensure data is written
con.close()
exit()

# We want to do only some tables at a time
# tables = query.keys()
tables = ["DR3_FINALCUT_IMAGE_FILEPATH", "DR3_FINALCUT_CATALOG_FILEPATH",
          "DR3_COADD_IMAGE_FILEPATH", "DR3_COADD_CATALOG_FILEPATH"]
# tables = ["DR3_FINALCUT_CATALOG_FILEPATH"]
# tables = ["DR3_FINALCUT_IMAGE_FILEPATH"]

# In case we want to do this from existing parquet tables
con = duckdb.connect('decade_metadata.duckdb')
for parquet_name in tables:
    t0 = time.time()
    print(f"-- Will load parquet table: {parquet_name}")
    df = pd.read_parquet(f"{parquet_name}.parquet")
    print(f"-- Done loading in: {elapsed_time(t0)}[s]")
    if 'PATH' in df.columns:
        df['PATH'] = df['PATH'].str.replace('DEC_Taiga/', 'DEC/', regex=False)
        df['PATH'] = df['PATH'].str.replace('ACT_Taiga/', 'ACT/', regex=False)
    con.register('df_view', df)
    t1 = time.time()
    q = f"CREATE OR REPLACE TABLE {parquet_name} AS SELECT * FROM df_view"
    print(f"-- Will run: \n{q}\n")
    con.execute(q)
    con.unregister('df_view')
    print(f"-- Done inserting in: {elapsed_time(t1)}[s]")
    print(f"-- Total time {parquet_name}: {elapsed_time(t0)}[s]")

con.close()
