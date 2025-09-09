#!/usr/bin/env python

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


# Full DB
dbname = 'desdecade_full_metadata.duckdb'
tables = ["DR3_COADD_CATALOG_FILEPATH",
          "DR3_COADD_IMAGE_FILEPATH",
          "DR3_COADD_TIFF_FILEPATH",
          "DR3_FINALCUT_CATALOG_FILEPATH",
          "DR3_FINALCUT_IMAGE_FILEPATH",
          "Y6A2_COADD_CATALOG_FILEPATH",
          "Y6A2_COADD_IMAGE_FILEPATH",
          "Y6A2_COADD_TIFF_FILEPATH",
          "Y6A2_FINALCUT_CATALOG_FILEPATH",
          "Y6A2_FINALCUT_IMAGE_FILEPATH"]

# Now we make a duckDB table
# Connect to DuckDB persistent database (or use :memory:)
con = duckdb.connect(dbname)

for table in tables:

    # get the archive_root
    if "DR3" in table:
        archive_root = "/taiga/deca_archive/"
    elif "Y6A2" in table:
        archive_root = "/taiga/des_archive/"

    if "_IMAGE_" in table:
        query = f"select PATH, FILENAME, COMPRESSION from {table}"
        df = con.execute(query).fetchdf()
        df['FILE'] = archive_root + df['PATH'] + "/" + df['FILENAME'] + df['COMPRESSION']
    else:
        query = f"select PATH, FILENAME from {table}"
        df = con.execute(query).fetchdf()
        df['FILE'] = archive_root + df['PATH'] + "/" + df['FILENAME']

    # df = pd.read_sql(query, con)
    df['FILE'].to_csv(f"{table}.filenames", index=False, header=False)
    print(f"Wrote {table}.filenames")
