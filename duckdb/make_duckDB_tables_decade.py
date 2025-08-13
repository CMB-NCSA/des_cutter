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
query['DR3_COADD_FILEPATH'] = """
select c.FILENAME, c.TILENAME, c.BAND, f.PATH, f.COMPRESSION,
       t.CROSSRA0, t.RACMIN, t.RACMAX, t.DECCMIN, t.DECCMAX,
       t.RA_CENT, t.DEC_CENT, t.RA_SIZE, t.DEC_SIZE,
       t.RAC1, t.RAC2, t.RAC3, t.RAC4, t.DECC1, t.DECC2, t.DECC3, t.DECC4
 from COADD c, PROCTAG p, FILE_ARCHIVE_INFO f, COADDTILE_GEOM t
  where p.TAG = 'DR3_COADD'
    and t.tilename = c.TILENAME
    and c.FILETYPE = 'coadd'
    and p.PFW_ATTEMPT_ID = c.PFW_ATTEMPT_ID
    and f.FILENAME = c.FILENAME"""

for parquet_name, q in query.items():
    print(f"Will run query: {q}")
    t0 = time.time()
    df = pd.read_sql(q, dbh)
    print(f"Done with query in {elapsed_time(t0)}[s]")
    df.to_parquet(f"{parquet_name}.parquet", engine="pyarrow", compression="snappy", index=True)

con = duckdb.connect("decade_metadata.duckdb")
for parquet_name, q in query.items():
    t0 = time.time()
    query = f"CREATE TABLE {parquet_name} AS SELECT * FROM '{parquet_name}.parquet'"
    con.execute(query)
    print(f"Wrote DuckDB table: {parquet_name} in {elapsed_time(t0)}[s]")

con.execute("VACUUM")  # Ensure data is written
con.close()
