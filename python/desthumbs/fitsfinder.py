import numpy
import logging
import os
import oracledb
import configparser

XSIZE_default = 10.0
YSIZE_default = 10.0

logger = logging.getLogger(__name__)


def load_db_config(config_file, profile):
    config = configparser.ConfigParser()
    config.read(config_file)

    section = dict(config[profile])
    section['dsn'] = f'{section["server"]}:{section["port"]}/{section["name"]}'
    return section


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

def connect_db(user=None, password=None, dsn=None):
    """
    Establish OracleDB connection using environment variables or passed-in values.
    """
    user = user or os.getenv('DESDB_USER')
    password = password or os.getenv('DESDB_PASS')
    dsn = dsn or os.getenv('DESDB_DSN')  # e.g., 'db-des.ncsa.illinois.edu:1521/desoper'

    logger.info(f"Connecting to OracleDB as user={user}, dsn={dsn}")
    try:
        con = oracledb.connect(user=user, password=password, dsn=dsn)
        return con
    except oracledb.Error as e:
        logger.error(f"Failed to connect to OracleDB: {e}")
        raise


def check_xysize(df, xsize=None, ysize=None):
    """
    Check if xsize/ysize are set from command-line or read from input CSV.
    """
    nobj = len(df.RA.values)
    if xsize:
        xsize = numpy.array([xsize] * nobj)
    else:
        try:
            xsize = df.XSIZE.values
        except Exception:
            xsize = numpy.array([XSIZE_default] * nobj)

    if ysize:
        ysize = numpy.array([ysize] * nobj)
    else:
        try:
            ysize = df.YSIZE.values
        except Exception:
            ysize = numpy.array([YSIZE_default] * nobj)

    return xsize, ysize


def get_query(tablename, bands=None, filetypes=None, date_start=None, date_end=None, yearly=None):
    #Builds the SQL query string for retrieving the metadata from OracleDB
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


def query2rec(query, dbhandle):
    #Performs the SQL query and returns the numpy recarray
    cur = dbhandle.cursor()
    cur.execute(query)
    tuples = cur.fetchall()

    if tuples:
        names = [d[0] for d in cur.description]
        return numpy.rec.array(tuples, names=names)
    else:
        logger.error("# DB Query in query2rec() returned no results")
        raise RuntimeError(f"# Error with query: {query}")
