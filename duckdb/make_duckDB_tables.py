#!/usr/bin/env python

import configparser
import oracledb
import os
import pandas as pd
import duckdb
import time


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


# db_section = 'db-dessci'
# schema = 'des_admin'

db_section = 'db-desoper'
schema = 'prod'
config_file = os.path.join(os.environ['HOME'], 'dbconfig.ini')
# Get the connection credentials and information
creds = load_db_config(config_file, db_section)
dbh = oracledb.connect(user=creds['user'],
                       password=creds['passwd'],
                       dsn=creds['dsn'])


# t0 = time.time()
# df = pd.read_sql(query, dbh)
# print("Done reading table")
# df.to_parquet(f"{parquet_name}.parquet", engine="pyarrow", compression="snappy", index=True)
# print(f"Done: {parquet_name} in {elapsed_time(t0)}[s]")
# exit()


query = {}
parquet_names = ["Y6A2_COADD_FILEPATH", "Y6A2_FINALCUT_FILEPATH", "Y6A2_COADDTILE_GEOM"]
# 1 Query to create Y6A2_COADD_FILEPATH parquet file
query["Y6A2_COADD_FILEPATH"] = """
  select c.FILENAME, c.TILENAME, c.BAND, c.FILETYPE, f.PATH, f.COMPRESSION, d.CREATED_DATE, d.FILESIZE, d.MD5SUM
  from prod.COADD c, prod.PROCTAG, prod.FILE_ARCHIVE_INFO f, prod.DESFILE d
         where prod.PROCTAG.TAG='Y6A2_COADD'
           and c.PFW_ATTEMPT_ID=prod.PROCTAG.PFW_ATTEMPT_ID
           and f.FILENAME=c.FILENAME
           and c.FILENAME=d.FILENAME
           and f.COMPRESSION=d.COMPRESSION"""

# 2. Query to create Y2A2_FINALCUT_FILEPATH
query["Y6A2_FINALCUT_FILEPATH"] = """
select i.FILENAME, f.PATH, i.BAND, i.EXPTIME, i.AIRMASS, i.FWHM, i.NITE, i.EXPNUM, i.CCDNUM, e.DATE_OBS, e.MJD_OBS,
       d.CREATED_DATE, d.FILESIZE, d.MD5SUM,
       i.CROSSRA0, i.RACMIN, i.RACMAX, i.DECCMIN, i.DECCMAX,
       i.RAC1, i.RAC2, i.RAC3, i.RAC4, i.DECC1, i.DECC2, i.DECC3, i.DECC4
from prod.IMAGE i, prod.EXPOSURE e, prod.FILE_ARCHIVE_INFO f, prod.PROCTAG, prod.DESFILE d
      where prod.PROCTAG.TAG='Y6A1_FINALCUT'
        and i.EXPNUM=e.EXPNUM
        and i.FILENAME=f.FILENAME
        and i.FILENAME=d.FILENAME
        and i.FILETYPE='red_immask'
        and f.COMPRESSION=d.COMPRESSION
"""

# 3. Query to get the coadd geometry
query["Y6A2_COADDTILE_GEOM"] = "SELECT * FROM COADDTILE_GEOM"

# Loop over parquet_names to create parquet tables
for parquet_name in parquet_names:
    t0 = time.time()
    df = pd.read_sql(query[parquet_name], dbh)
    print("Done reading table")
    df.to_parquet(f"{parquet_name}.parquet", engine="pyarrow", compression="snappy", index=True)
    print(f"Done: {parquet_name} in {elapsed_time(t0)}[s]")

# Now we make a duckDB DB in the filesystem
# Connect to DuckDB persistent database (or use :memory:)
con = duckdb.connect("des_metadata.duckdb")
for parquet_name in parquet_names:
    t0 = time.time()
    query = f"CREATE TABLE {parquet_name} AS SELECT * FROM '{parquet_name}.parquet'"
    con.execute(query)
    print(f"Wrote DuckDB table: {parquet_name} in {elapsed_time(t0)}[s]")

con.execute("VACUUM")  # Ensure data is written
con.close()
exit()

# old way of doing this on dessci

oracle2parquet_names = {
    # Notice change of name from Y6A1_COADDTILE_GEOM --> Y6A2_COADDTILE_GEOM
    'des_admin.Y6A1_COADDTILE_GEOM': 'Y6A2_COADDTILE_GEOM',
    'felipe.Y6A2_COADD_FILEPATH': 'Y6A2_COADD_FILEPATH',
    'felipe.Y6A2_FINALCUT_FILEPATH': 'Y6A2_FINALCUT_FILEPATH',
}

# Loop over all tables and create .parquet files for each one
for oracle_name, parquet_name in oracle2parquet_names.items():
    t0 = time.time()
    query = f"SELECT * FROM {oracle_name}"
    df = pd.read_sql(query, dbh)
    print("Done reading table")
    df.to_parquet(f"{parquet_name}.parquet", engine="pyarrow", compression="snappy", index=True)
    print(f"Done: {parquet_name} in {elapsed_time(t0)}[s]")

# Now we make a duckDB DB in the filesystem
# Connect to DuckDB persistent database (or use :memory:)
con = duckdb.connect("des_metadata.duckdb")
for oracle_name, parquet_name in oracle2parquet_names.items():
    t0 = time.time()
    query = f"CREATE TABLE {parquet_name} AS SELECT * FROM '{parquet_name}.parquet'"
    con.execute(query)
    print(f"Wrote DuckDB table: {parquet_name} in {elapsed_time(t0)}[s]")

con.execute("VACUUM")  # Ensure data is written
con.close()
