import collections
import socket
import numpy
import os
import pandas
import warnings
import logging
import datetime
warnings.filterwarnings("ignore", category=UserWarning, message=".*pandas only supports SQLAlchemy.*")

# Logger
LOGGER = logging.getLogger(__name__)
logger = LOGGER

# Default xsize, ysize in arcmin
XSIZE_DEFAULT = 1.0
YSIZE_DEFAULT = 1.0


def check_columns(cols, req_cols):
    """ Test that all required columns are present"""
    for c in req_cols:
        if c not in cols:
            raise TypeError(f'column {c} in file')
    return


def check_xy(df, xsize=None, ysize=None):
    """
    Check if xsize/ysize are set from command-line or read from input CSV.
    """
    nobj = len(df)
    if xsize:
        xsize = numpy.array([xsize]*nobj)
    else:
        try:
            xsize = df.XSIZE.values
        except ValueError:
            xsize = numpy.array([XSIZE_DEFAULT]*nobj)

    if ysize:
        ysize = numpy.array([ysize]*nobj)
    else:
        try:
            ysize = df.YSIZE.values
        except ValueError:
            ysize = numpy.array([YSIZE_DEFAULT]*nobj)
    return xsize, ysize


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


def find_tilename_radec(ra, dec, con, tag='Y6A2'):
    """
    Find the DES coadd tile name that contains the given RA and DEC position.
    This function queries Oracle database to determine which coadd tile the sky coordinate (RA, DEC)
    falls into.
    """
    if ra < 0:
        exit("ERROR: Please provide RA>0 and RA<360")

    QUERY_TILENAME_RADEC = """
    select TILENAME from {TAG}_COADDTILE_GEOM
           where (CROSSRA0='N' AND ({RA} BETWEEN RACMIN and RACMAX) AND ({DEC} BETWEEN DECCMIN and DECCMAX)) OR
                 (CROSSRA0='Y' AND ({RA180} BETWEEN RACMIN-360 and RACMAX) AND ({DEC} BETWEEN DECCMIN and DECCMAX))
    """

    if ra > 180:
        ra180 = 360 - ra
    else:
        ra180 = ra
    query = QUERY_TILENAME_RADEC.format(RA=ra, DEC=dec, RA180=ra180, TAG=tag)
    tilenames_dict = query2dict_of_columns(query, con, array=False)

    if len(tilenames_dict) < 1:
        LOGGER.warning(f"No tile found at ra:{ra}, dec:{dec}\n")
        return False
    else:
        return tilenames_dict['TILENAME'][0]


def find_tilenames_radec(ra, dec, con, tag='Y6A2'):
    """
    Find the tilename for each ra,dec and bundle them as dictionaries per tilename
    """
    indices = {}
    tilenames = []
    tilenames_matched = []
    for k, (ra_val, dec_val) in enumerate(zip(ra, dec)):

        tilename = find_tilename_radec(ra_val, dec_val, con, tag=tag)
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


def find_finalcut_images(ra, dec, dbh, bands=None, date_start=None, date_end=None):
    """
    Find the FINALCUT images via sql query
    """
    results = []
    for k, (ra_val, dec_val) in enumerate(zip(ra, dec)):
        # Get the formatted query for ra, dec, dates and bands
        query = get_query_finalcut(ra_val, dec_val, bands=bands, date_start=date_start, date_end=date_end)
        LOGGER.debug(f"Will run query:\n{query}\n")
        # Load into a pandas df
        df = pandas.read_sql(query, con=dbh)
        if 'COMPRESSION' in df.columns:
            df['FILE'] = df['PATH'].astype(str) + '/' + df['FILENAME'].astype(str) + df['COMPRESSION']
        else:
            df['FILE'] = df['PATH'].astype(str) + '/' + df['FILENAME'].astype(str)
        results.append(df)
    return results


def get_query_finalcut(ra, dec, bands=None, date_start=None, date_end=None):

    query_FINALCUTFILES = """
    select FILENAME, COMPRESSION, PATH, BAND, EXPTIME, NITE, EXPNUM, DATE_OBS, MJD_OBS
    from Y6A2_FINALCUT_FILEPATH
      where
      ((CROSSRA0='N' AND ({RA} BETWEEN RACMIN and RACMAX) AND ({DEC} BETWEEN DECCMIN and DECCMAX)) OR
       (CROSSRA0='Y' AND ({RA} BETWEEN RACMIN-360 and RACMAX) AND ({DEC} BETWEEN DECCMIN and DECCMAX)))
      {and_bands}
      {and_dates}
      order by EXPNUM
    """

    # BAND formatting
    if bands == 'all' or bands is None:
        and_bands = ''
    else:
        in_bands = ','.join(f"'{s}'" for s in bands)
        and_bands = f"and BAND IN ({in_bands})"

    # DATE formatting
    if (isinstance(date_start, str) and isinstance(date_end, str)) \
       or (isinstance(date_start, datetime.date) and isinstance(date_end, datetime.date)):
        and_dates = f"and DATE_OBS BETWEEN '{date_start}' AND '{date_end}'"
    else:
        and_dates = ''

    query = query_FINALCUTFILES.format(
        RA=ra,
        DEC=dec,
        and_bands=and_bands,
        and_dates=and_dates)
    return query


def get_coaddfiles_tilename(tilename, dbh, bands='all'):
    """
    Build the query and get the coadd files for a TILENAME
    Replace to pandas dataframe
    """

    if bands == 'all':
        and_bands = ''
    else:
        sbands = "'" + "','".join(bands) + "'"  # trick to format
        and_bands = f"BAND in ({sbands}) and"

    QUERY_COADDFILES = """
    select FILENAME, TILENAME, BAND, FILETYPE, PATH, COMPRESSION
     from Y6A2_COADD_FILEPATH
            where
              FILETYPE='coadd' and
              {and_BANDS} TILENAME='{TILENAME}'"""

    query = QUERY_COADDFILES.format(TILENAME=tilename, and_BANDS=and_bands)
    LOGGER.info(f"Running query: {query}")
    rec = query2rec(query, dbh)
    # Return a record array with the query
    return rec


def get_archive_root():
    """Function retreives the archive root"""

    if 'DES_ARCHIVE_ROOT' in os.environ:
        archive_root = os.environ['DES_ARCHIVE_ROOT']
    else:
        # Try to auto-figure out from location
        address = socket.getfqdn()
        if address.find('cosmology.illinois.edu') >= 0:
            archive_root = '/archive_data/desarchive/OPS_Taiga/'
        elif address.find('spt3g') >= 0:
            archive_root = '/des_archive/'
        else:
            logger.warning(f"archive_root undefined for: {address}")
            archive_root = ''

    LOGGER.debug(f"Getting the archive root: {archive_root}\n")
    return archive_root
