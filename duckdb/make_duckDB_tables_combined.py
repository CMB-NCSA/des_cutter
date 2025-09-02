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


def make_db(tables, dbname):

    """
    Make duckdb database from existing parquet tables
    tables: list of parquet tables
    dbname: Name of the new/existing DB (filename)
    """
    t0 = time.time()
    con = duckdb.connect(dbname)
    for parquet_name in tables:
        t1 = time.time()
        print(f"Will load parquet table: {parquet_name}")
        df = pd.read_parquet(f"{parquet_name}.parquet")
        print(f"Done loading in: {elapsed_time(t1)}[s]")
        if 'PATH' in df.columns:
            df['PATH'] = df['PATH'].str.replace('DEC_Taiga/', 'DEC/', regex=False)
            df['PATH'] = df['PATH'].str.replace('ACT_Taiga/', 'ACT/', regex=False)
            df['PATH'] = df['PATH'].str.replace('OPS_Taiga/', '', regex=False)

        con.register('df_view', df)
        t2 = time.time()
        q = f"CREATE OR REPLACE TABLE {parquet_name} AS SELECT * FROM df_view"
        print(f"Will run: \n{q}\n")
        con.execute(q)
        con.unregister('df_view')
        print(f"Done inserting in: {elapsed_time(t2)}[s]")
        print(f"Total {parquet_name} time: {elapsed_time(t1)}[s]")
    con.close()
    print(f"Total time for DB: {dbname} time: {elapsed_time(t0)}[s]")


# Lite DB
tables = ["Y6A2_COADDTILE_GEOM",
          "Y6A2_COADD_IMAGE_FILEPATH",
          "Y6A2_FINALCUT_IMAGE_FILEPATH",
          "DR3_COADDTILE_GEOM",
          "DR3_COADD_IMAGE_FILEPATH",
          "DR3_FINALCUT_IMAGE_FILEPATH"]
dbname = 'desdecade_lite_metadata.duckdb'
make_db(tables, dbname)

# Full DB
tables = ["DR3_COADDTILE_GEOM",
          "DR3_COADD_CATALOG_FILEPATH",
          "DR3_COADD_IMAGE_FILEPATH",
          "DR3_COADD_TIFF_FILEPATH",
          "DR3_FINALCUT_CATALOG_FILEPATH",
          "DR3_FINALCUT_IMAGE_FILEPATH",
          "Y6A2_COADDTILE_GEOM",
          "Y6A2_COADD_CATALOG_FILEPATH",
          "Y6A2_COADD_IMAGE_FILEPATH",
          "Y6A2_COADD_TIFF_FILEPATH",
          "Y6A2_FINALCUT_CATALOG_FILEPATH",
          "Y6A2_FINALCUT_IMAGE_FILEPATH"]
dbname = 'desdecade_full_metadata.duckdb'
make_db(tables, dbname)
