#!/usr/bin/env python

import os
import pandas as pd
import oracledb
import configparser
import time
import glob
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


# Connect
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

parquet_name = 'CATALOG_r'
query = """
select c.FILENAME, f.PATH, c.FILETYPE, c.BAND, c.CCDNUM, c.PFW_ATTEMPT_ID
 from CATALOG c, FILE_ARCHIVE_INFO f, PROCTAG p
  where p.TAG = 'DR3_FINALCUT'
    and p.pfw_attempt_id = c.pfw_attempt_id
    and f.FILENAME=c.FILENAME
    and c.FILETYPE='cat_finalcut'
    and c.BAND='r' and ROWNUM < 100000
"""

t0 = time.time()
df = pd.read_sql(query, dbh)
print(f"-- Done with query in {elapsed_time(t0)}[s]")
print(df)
df.to_parquet(f"{parquet_name}.parquet", engine="pyarrow", compression="snappy", index=True)
