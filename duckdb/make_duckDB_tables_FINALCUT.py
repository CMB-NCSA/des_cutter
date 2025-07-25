#!/usr/bin/env python

import configparser
import oracledb
import os
import pandas as pd
import duckdb
import time
from tqdm import tqdm  # progress bar


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


# -- The names of the tables
oracle_name = 'felipe.Y6A2_FINALCUT_FILEPATH'
parquet_name = 'Y6A2_FINALCUT_FILEPATH'

# --- Step 1: Connect to first DB and get initial DataFrame
db_section = 'db-dessci'
schema = 'des_admin'
config_file = os.path.join(os.environ['HOME'], 'dbconfig.ini')
# Get the connection credentials and information
creds = load_db_config(config_file, db_section)
dbh_desci = oracledb.connect(user=creds['user'],
                             password=creds['passwd'],
                             dsn=creds['dsn'])

t0 = time.time()
# Fetch a small sample (first 100 rows) for testing
query = f"SELECT * FROM {oracle_name} FETCH FIRST 200000 ROWS ONLY"
# query = f"SELECT * FROM {oracle_name}"
df = pd.read_sql(query, dbh_desci)
print(f"Done reading table {oracle_name} in {elapsed_time(t0)}[s]")

# --- Step 2: Prepare new columns (empty)
df['CREATED_DATE'] = None
df['FILESIZE'] = None
df['MD5SUM'] = None

# --- Step 3: Connect to second DB (where prod.DESFILE lives)
db_section = 'db-desoper'
schema = 'prod'
creds = load_db_config(config_file, db_section)
dbh_desoper = oracledb.connect(user=creds['user'],
                               password=creds['passwd'],
                               dsn=creds['dsn'])
cur = dbh_desoper.cursor()


# --- Step 4: Batched query loop (collect results)
batch_size = 1000  # adjust as needed
all_filenames = df['FILENAME'].tolist()
results = []  # store all rows from DESFILE here

for start in tqdm(range(0, len(all_filenames), batch_size), desc="Fetching DESFILE info"):
    batch = all_filenames[start:start + batch_size]

    placeholders = ",".join(f":{i+1}" for i in range(len(batch)))
    query2 = f"""
        SELECT FILENAME, CREATED_DATE, FILESIZE, MD5SUM
        FROM prod.DESFILE
        WHERE FILENAME IN ({placeholders})
    """
    cur.execute(query2, batch)
    results.extend(cur.fetchall())

# --- Step 5: Merge results into df (once, at the end)
if results:
    df_extra = pd.DataFrame(results, columns=['FILENAME', 'CREATED_DATE', 'FILESIZE', 'MD5SUM'])
    df = df.merge(df_extra, on='FILENAME', how='left', suffixes=('', '_new'))

    # Move new data into the main columns
    for col in ['CREATED_DATE', 'FILESIZE', 'MD5SUM']:
        df[col] = df[f"{col}_new"].combine_first(df[col])
        df.drop(columns=[f"{col}_new"], inplace=True)

# --- df now contains all data
print(df.head())


print("Done inserting to table")
df.to_parquet(f"{parquet_name}.parquet", engine="pyarrow", compression="snappy", index=True)
print(f"Done: {parquet_name} in {elapsed_time(t0)}[s]")
exit()

con = duckdb.connect("des_metadata.duckdb")
query = f"CREATE TABLE {parquet_name} AS SELECT * FROM '{parquet_name}.parquet'"


exit()
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
