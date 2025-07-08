#!/usr/bin/env python

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


# Now we make a duckDB table
# Connect to DuckDB persistent database (or use :memory:)
con = duckdb.connect("des_metadata.duckdb")


# Test queries
query = """
SELECT TILENAME FROM Y6A2_COADDTILE_GEOM
WHERE
    (CROSSRA0 = 'N' AND (0.29782658 BETWEEN RACMIN AND RACMAX) AND (0.029086056 BETWEEN DECCMIN AND DECCMAX)) OR
    (CROSSRA0 = 'Y' AND ((0.29782658 BETWEEN RACMIN - 360 AND RACMAX)) AND (0.029086056 BETWEEN DECCMIN AND DECCMAX))
"""
t0 = time.time()
results = con.execute(query).fetchdf()
print(results)
print(f"Done in {elapsed_time(t0)}[s]")

query = """
select FILENAME, TILENAME, BAND, FILETYPE, PATH, COMPRESSION
from Y6A2_COADD_FILEPATH
where
    FILETYPE='coadd' and
    TILENAME='DES2359+0001'
"""
t0 = time.time()
results = con.execute(query).fetchdf()
print(results)
print(f"Done in {elapsed_time(t0)}[s]")

query = """
select FILENAME, PATH, COMPRESSION, BAND, EXPTIME, NITE, EXPNUM, DATE_OBS, MJD_OBS from Y6A2_FINALCUT_FILEPATH
  where ((CROSSRA0='N' AND (0.29782658 BETWEEN RACMIN and RACMAX) AND (0.029086056 BETWEEN DECCMIN and DECCMAX)) OR
  (CROSSRA0='Y' AND (0.29782658 BETWEEN RACMIN-360 and RACMAX) AND (0.029086056 BETWEEN DECCMIN and DECCMAX)))
  and DATE_OBS between '2013-10-25T00:05:49' and '2014-12-25T00:05:49'
  and BAND='r' order by EXPNUM
"""
t0 = time.time()
results = con.execute(query).fetchdf()
print(results)
print(f"Done in {elapsed_time(t0)}[s]")
