#!/usr/bin/env python

import configparser
import oracledb
import os
import pandas as pd
import numpy
import collections
import time

import sys
SOUT = sys.stdout


def load_db_config(config_file, profile):
    config = configparser.ConfigParser()
    config.read(config_file)

    section = dict(config[profile])
    section['dsn'] = f'{section["server"]}:{section["port"]}/{section["name"]}'
    return section


def check_columns(cols, req_cols):
    """ Test that all required columns are present"""
    for c in req_cols:
        if c not in cols:
            raise TypeError('column %s in file' % c)
    return


def fix_compression(rec):
    """
    Here we fix 'COMPRESSION from None --> '' if present
    """
    if rec is False:
        pass
    elif 'COMPRESSION' in rec.dtype.names:
        compression = ['' if c is None else c for c in rec['COMPRESSION']]
        rec['COMPRESSION'] = numpy.array(compression)
    return rec


def get_archive_root(dbh, schema='prod', verb=False):

    if schema != 'prod':
        archive_root = "/archive_data/desarchive"
        return archive_root

    QUERY_ARCHIVE_ROOT = {}

    name = {}
    name['prod'] = 'desar2home'
    QUERY_ARCHIVE_ROOT['prod'] = "select root from prod.ops_archive where name='%s'" % name['prod']
    if verb:
        SOUT.write("# Getting the archive root name for section: %s\n" % name[schema])
        SOUT.write("# Will execute the SQL query:\n********\n %s\n********\n" % QUERY_ARCHIVE_ROOT[schema])
    cur = dbh.cursor()
    cur.execute(QUERY_ARCHIVE_ROOT[schema])
    archive_root = cur.fetchone()[0]
    cur.close()
    return archive_root


def query2dict_of_columns(query, dbhandle, array=False):
    """
    Transforms the result of an SQL query and a Database handle object [dhandle]
    into a dictionary of list or numpy arrays if array=True
    """

    # Get the cursor from the DB handle
    cur = dbhandle.cursor()
    # Execute
    cur.execute(query)
    # Get them all at once
    list_of_tuples = cur.fetchall()
    # Get the description of the columns to make the dictionary
    desc = [d[0] for d in cur.description]

    querydic = collections.OrderedDict()  # We will populate this one
    cols = list(zip(*list_of_tuples))
    for k, val in enumerate(cols):
        key = desc[k]
        if array:
            if isinstance(val[0], str):
                querydic[key] = numpy.array(val, dtype=object)
            else:
                querydic[key] = numpy.array(val)
        else:
            querydic[key] = cols[k]
    return querydic


def query2rec(query, dbhandle, verb=False):
    """
    Queries DB and returns results as a numpy recarray.
    """
    # Get the cursor from the DB handle
    cur = dbhandle.cursor()
    # Execute
    cur.execute(query)
    tuples = cur.fetchall()

    # Return rec array
    if tuples:
        names = [d[0] for d in cur.description]
        return numpy.rec.array(tuples, names=names)

    if verb:
        print("# WARNING DB Query in query2rec() returned no results")
    return False


def find_tilename_radec(ra, dec, dbh, schema='prod'):

    if ra < 0:
        exit("ERROR: Please provide RA>0 and RA<360")

    if schema == "prod":
        tablename = "COADDTILE_GEOM"
    elif schema == "des_admin":
        tablename = "Y6A1_COADDTILE_GEOM"
    else:
        raise Exception(f"ERROR: COADDTILE table not defined for schema: {schema}")

    coaddtile_geom = f"{schema}.{tablename}"

    QUERY_TILENAME_RADEC = """
    select TILENAME from {COADDTILE_GEOM}
           where (CROSSRA0='N' AND ({RA} BETWEEN RACMIN and RACMAX) AND ({DEC} BETWEEN DECCMIN and DECCMAX)) OR
                 (CROSSRA0='Y' AND ({RA180} BETWEEN RACMIN-360 and RACMAX) AND ({DEC} BETWEEN DECCMIN and DECCMAX))
    """

    if ra > 180:
        ra180 = 360 - ra
    else:
        ra180 = ra
    query = QUERY_TILENAME_RADEC.format(RA=ra, DEC=dec, RA180=ra180, COADDTILE_GEOM=coaddtile_geom)
    tilenames_dict = query2dict_of_columns(query, dbh, array=False)

    if len(tilenames_dict) < 1:
        SOUT.write("# WARNING: No tile found at ra:%s, dec:%s\n" % (ra, dec))
        return False
    else:
        return tilenames_dict['TILENAME'][0]
    return

#fitsfinder
def find_tilenames_radec(ra, dec, dbh, schema='prod'):

    """
    Find the tilename for each ra,dec and bundle them as dictionaries per tilename
    """

    indices = {}
    tilenames = []
    tilenames_matched = []
    for k in range(len(ra)):

        tilename = find_tilename_radec(ra[k], dec[k], dbh, schema=schema)
        tilenames_matched.append(tilename)

        # Write out the results
        if not tilename:  # No tilename found
            # Here we could do something to store the failed (ra,dec) pairs
            continue

        # Store unique values and initialize list of indices grouped by tilename
        if tilename not in tilenames:
            indices[tilename] = []
            tilenames.append(tilename)

        indices[tilename].append(k)

    return tilenames, indices, tilenames_matched


def get_coaddfiles_tilename_bytag_old(tilename, dbh, tag, bands='all', schema='prod'):

    QUERY_COADDFILES_BANDS = {}
    QUERY_COADDFILES_ALL = {}

    QUERY_COADDFILES_BANDS['des_admin'] = """
    select c.FILENAME, c.TILENAME, c.BAND, f.PATH, f.COMPRESSION
     from des_admin.{TAG}_COADD c, des_admin.{TAG}_FILE_ARCHIVE_INFO f
            where
              c.FILETYPE='coadd' and
              c.BAND in ({BANDS}) and
              f.FILENAME=c.FILENAME and
              c.TILENAME='{TILENAME}'"""

    QUERY_COADDFILES_ALL['des_admin'] = """
    select c.FILENAME, c.TILENAME, c.BAND, f.PATH, f.COMPRESSION
     from des_admin.{TAG}_COADD c, des_admin.{TAG}_FILE_ARCHIVE_INFO f
            where
              c.FILETYPE='coadd' and
              f.FILENAME=c.FILENAME and
              c.TILENAME='{TILENAME}'"""

    QUERY_COADDFILES_BANDS['prod'] = """
    select c.FILENAME, c.TILENAME, c.BAND, f.PATH, f.COMPRESSION
     from prod.COADD c, prod.PROCTAG, prod.FILE_ARCHIVE_INFO f
            where
              c.FILETYPE='coadd' and
              c.BAND in ({BANDS}) and
              prod.PROCTAG.TAG='{TAG}_COADD' and
              c.PFW_ATTEMPT_ID=PROCTAG.PFW_ATTEMPT_ID and
              f.FILENAME=c.FILENAME and
              c.TILENAME='{TILENAME}'"""

    QUERY_COADDFILES_ALL['prod'] = """
    select c.FILENAME, c.TILENAME, c.BAND, f.PATH, f.COMPRESSION
     from prod.COADD c, prod.PROCTAG, prod.FILE_ARCHIVE_INFO f
            where prod.PROCTAG.TAG='{TAG}_COADD' and
              c.FILETYPE='coadd' and
              c.PFW_ATTEMPT_ID=PROCTAG.PFW_ATTEMPT_ID and
              f.FILENAME=c.FILENAME and
              c.TILENAME='{TILENAME}'"""

    if bands == 'all':
        query = QUERY_COADDFILES_ALL[schema].format(TILENAME=tilename, TAG=tag)
    else:
        sbands = "'" + "','".join(bands) + "'"  # trick to format
        query = QUERY_COADDFILES_BANDS[schema].format(TILENAME=tilename, TAG=tag, BANDS=sbands)

    print(query)
    rec = query2rec(query, dbh)
    # Return a record array with the query
    return rec


def get_coaddfiles_tilename_bytag(tilename, dbh, tag, bands='all'):

    if bands == 'all':
        and_BANDS = ''
    else:
        sbands = "'" + "','".join(bands) + "'"  # trick to format
        and_BANDS = "BAND in ({BANDS}) and".format(BANDS=sbands)

    QUERY_COADDFILES = """
    select FILENAME, TILENAME, BAND, FILETYPE, PATH, COMPRESSION
     from felipe.{TAG}_COADD_FILEPATH
            where
              FILETYPE='coadd' and
              {and_BANDS} TILENAME='{TILENAME}'"""

    query = QUERY_COADDFILES.format(TILENAME=tilename, TAG=tag, and_BANDS=and_BANDS)
    print(query)
    rec = query2rec(query, dbh)
    # Return a record array with the query
    return rec


def get_base_names(tilenames, ra, dec, prefix='DES'):
    names = []
    for k in range(len(ra)):
        if tilenames[k]:
            # name = get_thumbBaseName(ra[k], dec[k], prefix=prefix)
            name = f"{prefix}_{ra[k]}-{dec[k]}"
        else:
            name = False
        names.append(name)
    return names


if __name__ == "__main__":

    sout = sys.stdout

    db_section = 'db-desoper'
    schema = 'prod'

    # db_section = 'db-dessci'
    schema = 'des_admin'

    tag = "Y6A2"
    bands = 'all'
    # bands = ['g', 'r', 'z']

    # Read in CSV file with pandas
    inputList = 'bin/example_input_radec.csv'
    df = pd.read_csv(inputList)

    # Decide if we do search by RA,DEC or by COADD_ID
    searchbyID = False
    ra = df.RA.values
    dec = df.DEC.values
    nobj = len(ra)
    req_cols = ['RA', 'DEC']

    # Check columns for consistency
    check_columns(df.columns, req_cols)

    # Check the xsize and ysizes
    # xsize, ysize = check_xysize(df, args, nobj)

    #move this into new makeDESthumbslib
    
    config_file = os.path.join(os.environ['HOME'], 'dbconfig.ini')
    # Get the connection credentials and information
    creds = load_db_config(config_file, db_section)
    dbh = oracledb.connect(user=creds['user'],
                           password=creds['passwd'],
                           dsn=creds['dsn'])

    archive_root = get_archive_root(dbh, schema=schema, verb=True)
    print(archive_root)

    tilenames, indices, tilenames_matched = find_tilenames_radec(ra, dec, dbh, schema=schema)

    # Add them back to pandas dataframe and write a file
    df['TILENAME'] = tilenames_matched
    # Get the thumbname base names and the them the pandas dataframe too
    df['THUMBNAME'] = get_base_names(tilenames_matched, ra, dec, prefix="DES")
    matched_list = os.path.join(".", 'matched_'+os.path.basename(inputList))
    df.to_csv(matched_list, index=False)
    sout.write("# Wrote matched tilenames list to: %s\n" % matched_list)

    # Loop over all of the tilenames
    t0 = time.time()
    Ntile = 0
    for tilename in tilenames:

        t1 = time.time()
        Ntile = Ntile+1
        sout.write("# ----------------------------------------------------\n")
        sout.write("# Doing: %s [%s/%s]\n" % (tilename, Ntile, len(tilenames)))
        sout.write("# ----------------------------------------------------\n")

        # 1. Get all of the filenames for a given tilename
        filenames = get_coaddfiles_tilename_bytag(tilename, dbh, tag, bands=bands)
        print(filenames)
        if filenames is False:
            sout.write(f"# Skipping: {tilename} -- not in TAG: {tag} \n")
            continue
        # Fix compression for SV1/Y2A1/Y3A1 releases
        else:
            filenames = fix_compression(filenames)

        indx = indices[tilename]
        avail_bands = filenames.BAND
        # 2. Loop over all of the filename -- We could use multi-processing
        p = {}
        n_filenames = len(avail_bands)
        for k in range(n_filenames):
            # Rebuild the full filename with COMPRESSION if present
            if 'COMPRESSION' in filenames.dtype.names:
                filename = os.path.join(archive_root, filenames.PATH[k], filenames.FILENAME[k])+filenames.COMPRESSION[k]
            else:
                filename = os.path.join(archive_root, filenames.PATH[k], filenames.FILENAME[k])
            ar = (filename, ra[indx], dec[indx])
            print(filename)
