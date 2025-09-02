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


def find_tilename_radec(ra, dec, con, tag=None):
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
    tilenames_df = pandas.read_sql(query, con)
    if len(tilenames_df) < 1:
        LOGGER.warning(f"No tile found at ra:{ra}, dec:{dec}\n")
        return False
    else:
        return tilenames_df['TILENAME'][0]


def find_tilenames_radec(ra, dec, con, tag=None):
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


def find_finalcut_images(ra, dec, dbh, tag=None, bands=None, date_start=None, date_end=None):
    """
    Find the FINALCUT images via sql query
    """
    results = []
    for k, (ra_val, dec_val) in enumerate(zip(ra, dec)):
        # Get the formatted query for ra, dec, dates and bands
        query = get_query_finalcut(ra_val, dec_val, tag=tag, bands=bands, date_start=date_start, date_end=date_end)
        LOGGER.debug(f"Will run query:\n{query}\n")
        # Load into a pandas df
        df = pandas.read_sql(query, con=dbh)
        if 'COMPRESSION' in df.columns:
            df['FILE'] = df['PATH'].astype(str) + '/' + df['FILENAME'].astype(str) + df['COMPRESSION']
        else:
            df['FILE'] = df['PATH'].astype(str) + '/' + df['FILENAME'].astype(str)
        results.append(df)
    return results


def get_query_finalcut(ra, dec, tag='Y6A2', bands=None, date_start=None, date_end=None):

    query_FINALCUTFILES = """
    select FILENAME, COMPRESSION, PATH, BAND, EXPTIME, NITE, EXPNUM, DATE_OBS, MJD_OBS, MAG_ZERO, SIGMA_MAG_ZERO
    from {TAG}_FINALCUT_IMAGE_FILEPATH
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
        TAG=tag,
        and_bands=and_bands,
        and_dates=and_dates)
    return query


def get_coaddfiles_tilename(tilename, dbh, tag='Y6A2', bands='all'):
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
     from {TAG}_COADD_IMAGE_FILEPATH
            where
              FILETYPE='coadd' and
              {and_BANDS} TILENAME='{TILENAME}'"""

    query = QUERY_COADDFILES.format(TILENAME=tilename, and_BANDS=and_bands, TAG=tag)
    LOGGER.info(f"Running query: {query}")
    rec = pandas.read_sql(query, dbh)
    if len(rec) < 1:
        LOGGER.warning(f"No coadd files found tilename: {tilename}")
        return False
    else:
        return rec


def get_archive_root(tag):
    """Function retreives the archive root"""

    if 'DES_ARCHIVE_ROOT' in os.environ:
        des_archive_root = os.environ['DES_ARCHIVE_ROOT']
        LOGGER.debug(f"Getting DES archive root: {des_archive_root}\n")

    if 'DECA_ARCHIVE_ROOT' in os.environ:
        deca_archive_root = os.environ['DECA_ARCHIVE_ROOT']
        LOGGER.debug(f"Getting DECA archive root: {deca_archive_root}\n")

    # Try to auto-figure out from location
    address = socket.getfqdn()
    if address.find('cosmology.illinois.edu') >= 0:
        des_archive_root = "/taiga/des_archive"
        deca_archive_root = "/taiga/deca_archive"
    elif address.find('spt3g') >= 0:
        des_archive_root = "/des_archive/"
        deca_archive_root = "/deca_archive/"
    else:
        des_archive_root = "/des_archive/"
        deca_archive_root = "/deca_archive/"
        logger.warning(f"archive_root undefined for: {address} -- will use defaults")

    if tag == 'Y6A2':
        LOGGER.debug(f"Getting des_archive root: {des_archive_root}\n")
        return des_archive_root
    if tag == 'DR3':
        LOGGER.debug(f"Getting deca_archive root: {deca_archive_root}\n")
        return deca_archive_root
