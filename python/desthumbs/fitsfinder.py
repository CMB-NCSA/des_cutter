"""This module handles fits queries and finding the images within the database"""

import logging
import os
import sys
import collections
import configparser

import oracledb
import numpy

SOUT = sys.stdout

XSIZE_DEFAULT = 10.0
YSIZE_DEFAULT = 10.0

logger = logging.getLogger(__name__)


def load_db_config(config_file, profile):
    """Function parses and formats config file"""
    config = configparser.ConfigParser()
    config.read(config_file)
    section = dict(config[profile])
    section['dsn'] = f'{section["server"]}:{section["port"]}/{section["name"]}'
    return section


def connect_db(user=None, password=None, dsn=None):
    """
    Establish OracleDB connection using environment variables or passed-in values.
    """
    user = user or os.getenv('DESDB_USER')
    password = password or os.getenv('DESDB_PASS')
    dsn = dsn or os.getenv('DESDB_DSN')  # e.g., 'db-des.ncsa.illinois.edu:1521/desoper'

    logger.info("Connecting to OracleDB as user=%s, dsn=%s", user, dsn)
    try:
        con = oracledb.connect(user=user, password=password, dsn=dsn)
        return con
    except oracledb.Error as e:
        logger.error("Failed to connect to OracleDB: %s", e)
        raise


def check_columns(cols, req_cols):
    """ Test that all required columns are present"""
    for c in req_cols:
        if c not in cols:
            raise TypeError(f'column {c} in file')
    return


def check_xysize(df, args, nobj):
    """
    Check if xsize/ysize are set from command-line or read from input CSV.
    """
    if args.xsize:
        xsize = numpy.array([args.xsize]*nobj)
    else:
        try:
            xsize = df.XSIZE.values
        except ValueError:
            xsize = numpy.array([XSIZE_DEFAULT]*nobj)

    if args.ysize:
        ysize = numpy.array([args.ysize]*nobj)
    else:
        try:
            ysize = df.YSIZE.values
        except ValueError:
            ysize = numpy.array([YSIZE_DEFAULT]*nobj)
    return xsize, ysize


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


def query2dict_of_columns(query, con, array=False):
    """
    Transforms the result of an SQL query and a Database handle object [dhandle]
    into a dictionary of list or numpy arrays if array=True
    """
    # Get the cursor from the DB handle
    # cur = dbhandle.cursor() 
    # # Execute
    result = con.execute(query)
    # Get them all at once
    list_of_tuples = result.fetchall()
    # Get the description of the columns to make the dictionary
    desc = [d[0] for d in result.description]

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


def query2rec(query, dbhandle):
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
    logger.error("# DB Query in query2rec() returned no results")
    msg = f"# Error with query:{query}"
    raise RuntimeError(msg)


def find_tilename_radec(ra, dec, con):
    """
    Find the DES coadd tile name that contains the given RA and DEC position.
    This function queries Oracle database to determine which coadd tile the sky coordinate (RA, DEC)
    falls into.
    """
    if ra < 0:
        exit("ERROR: Please provide RA>0 and RA<360")

    # if schema == "prod":
    #     tablename = "COADDTILE_GEOM"
    # elif schema == "des_admin":
    #     tablename = "Y6A1_COADDTILE_GEOM"
    # else:
    #     raise ValueError(f"ERROR: COADDTILE table not defined for schema: {schema}")

    # coaddtile_geom = f"{schema}.{tablename}"

    QUERY_TILENAME_RADEC = """
    select TILENAME from Y6A2_COADDTILE_GEOM
           where (CROSSRA0='N' AND ({RA} BETWEEN RACMIN and RACMAX) AND ({DEC} BETWEEN DECCMIN and DECCMAX)) OR
                 (CROSSRA0='Y' AND ({RA180} BETWEEN RACMIN-360 and RACMAX) AND ({DEC} BETWEEN DECCMIN and DECCMAX))
    """

    if ra > 180:
        ra180 = 360 - ra
    else:
        ra180 = ra
    query = QUERY_TILENAME_RADEC.format(RA=ra, DEC=dec, RA180=ra180)
    tilenames_dict = query2dict_of_columns(query, con, array=False)

    if len(tilenames_dict) < 1:
        SOUT.write(f"# WARNING: No tile found at ra:{ra}, dec:{dec}\n")
        return False
    else:
        return tilenames_dict['TILENAME'][0]


def find_tilenames_radec(ra, dec, con): #schema
    """
    Find the tilename for each ra,dec and bundle them as dictionaries per tilename
    """
    
    indices = {}
    tilenames = []
    tilenames_matched = []
    for k, (ra_val, dec_val) in enumerate(zip(ra, dec)):

        tilename = find_tilename_radec(ra_val, dec_val, con)
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


def get_query(tablename, bands=None, filetypes=None, date_start=None, date_end=None, yearly=None):
    """Builds the SQL query string for retrieving the metadata from OracleDB"""
    query_files_template = """
    SELECT ID, FILEPATH || '/' || FILENAME AS FILE, BAND, DATE_BEG FROM {tablename}
      {where}
       {and_bands}
       {and_dates}
       {and_filetypes}
    """

    # BAND formatting
    if bands:
        in_bands = ','.join(f"'{s}'" for s in bands)
        and_bands = f"BAND IN ({in_bands})"
    else:
        and_bands = ''

    # FILETYPE formatting
    if filetypes:
        in_filetypes = ','.join(f"'{s}'" for s in filetypes)
        and_filetypes = f"FILETYPE IN ({in_filetypes})"
        if bands:
            and_filetypes = f"AND ({and_filetypes})"
    else:
        and_filetypes = ''

    # DATE formatting
    if isinstance(date_start, str) and isinstance(date_end, str):
        and_dates = f"DATE_BEG BETWEEN TO_DATE('{date_start}', 'YYYY-MM-DD') AND TO_DATE('{date_end}', 'YYYY-MM-DD')"
        and_dates_or = ' OR '
    else:
        and_dates = ''
        and_dates_or = ''

    # OBS_ID formatting
    if yearly:
        in_yearly = ','.join(f"'{s}'" for s in yearly)
        and_dates = f"{and_dates}{and_dates_or}OBS_ID IN ({in_yearly})"
    if bands or filetypes:
        and_dates = f"AND ({and_dates})" if and_dates else ''

    # Final WHERE clause
    where = 'WHERE' if and_bands or and_filetypes or and_dates else ''

    return query_files_template.format(
        tablename=tablename,
        where=where,
        and_bands=and_bands,
        and_filetypes=and_filetypes,
        and_dates=and_dates
    )


def get_coaddfiles_tilename_bytag(tilename, dbh, tag, bands='all'):
    if bands == 'all':
        and_bands = ''
    else:
        sbands = "'" + "','".join(bands) + "'"  # trick to format
        and_bands = f"BAND in ({sbands}) and"

    QUERY_COADDFILES = """
    select FILENAME, TILENAME, BAND, FILETYPE, PATH, COMPRESSION
     from felipe.{TAG}_COADD_FILEPATH
            where
              FILETYPE='coadd' and
              {and_BANDS} TILENAME='{TILENAME}'"""

    query = QUERY_COADDFILES.format(TILENAME=tilename, TAG=tag, and_BANDS=and_bands)
    print(query)
    rec = query2rec(query, dbh)
    # Return a record array with the query
    return rec


def get_archive_root(dbh, schema='prod', verb=False):
    """Function retreives the archive root"""
    if schema != 'prod':
        archive_root = "/archive_data/desarchive"
        return archive_root

    QUERY_ARCHIVE_ROOT = {}

    name = {}
    name['prod'] = 'desar2home'
    QUERY_ARCHIVE_ROOT['prod'] = f"select root from prod.ops_archive where name={name['prod']}"
    if verb:
        SOUT.write(f"# Getting the archive root name for section: {name[schema]}\n")
        SOUT.write(f"# Will execute the SQL query:\n********\n {QUERY_ARCHIVE_ROOT[schema]}\n********\n")
    cur = dbh.cursor()
    cur.execute(QUERY_ARCHIVE_ROOT[schema])
    archive_root = cur.fetchone()[0]
    cur.close()
    return archive_root
