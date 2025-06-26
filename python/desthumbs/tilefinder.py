
"""
Set of functions to find Tilenames and Filenames in the archive (old
Schema) for a set of positions (ra,dec)
F. Menanteau, NCSA July 2015
"""

import numpy
import despyastro
import sys
SOUT = sys.stdout

def get_archive_root(dbh,schema='prod',verb=False):

    QUERY_ARCHIVE_ROOT = {}

    name = {}
    name['des_admin'] = 'desardata'
    name['prod']   = 'desar2home'

    if schema == 'dr1':
        archive_root = '/des004/despublic/dr1_tiles/'
        return archive_root
        
    QUERY_ARCHIVE_ROOT['des_admin'] = "select archive_root from des_admin.archive_sites where location_name='%s'"  % name['des_admin']
    QUERY_ARCHIVE_ROOT['prod'] = "select root from prod.ops_archive where name='%s'" % name['prod'] 
    if verb:
        SOUT.write("# Getting the archive root name for section: %s\n" % name[schema])
        SOUT.write("# Will execute the SQL query:\n********\n** %s\n********\n" %  QUERY_ARCHIVE_ROOT[schema])
    cur = dbh.cursor()
    cur.execute(QUERY_ARCHIVE_ROOT[schema])
    archive_root = cur.fetchone()[0]
    cur.close()
    return archive_root


def find_tilename_id(id,tablename,dbh,schema='prod'):

    QUERY_TILENAME_ID = {}
    QUERY_TILENAME_ID['des_admin'] = """
    select TILENAME,RA,DEC from des_admin.{TABLENAME}@dessci
           where COADD_OBJECTS_ID={ID}"""
    
    QUERY_TILENAME_ID['prod'] = """
    select TILENAME,ALPHAWIN_J2000 as RA,DELTAWIN_J2000 as DEC, from prod.{TABLENAME}
           where ID={ID}"""

    tilenames_dict = despyastro.query2dict_of_columns(QUERY_TILENAME_ID[schema].format(ID=id,TABLENAME=tablename),dbh,array=False)
    if len(tilenames_dict)<1:
        SOUT.write("# WARNING: No tile found at ra:%s, dec:%s\n" % (ra,dec))
        return False
    else:
        return tilenames_dict['TILENAME'][0],tilenames_dict['RA'][0],tilenames_dict['DEC'][0]
    return


def find_tilenames_id(id,tablename,dbh,schema='prod'):

    import numpy

    """
    Find the tilename for each id and bundle them as dictionaries per tilename
    """

    indices = {}
    tilenames = []
    tilenames_matched = []
    ras = []
    decs = []
    for k in range(len(id)):
        
        tilename,ra,dec = find_tilename_id(id[k],tablename,dbh,schema=schema)
        tilenames_matched.append(tilename)
        if not tilename: # No tilename found
            # Here we could do something to store the failed (ra,dec) pairs
            continue
        ras.append(ra)
        decs.append(dec)
        # Store unique values and initialize list of indices grouped by tilename
        if tilename not in tilenames:
            indices[tilename]  = []
            tilenames.append(tilename)

        indices[tilename].append(k)

    return tilenames, numpy.array(ras), numpy.array(decs), indices, tilenames_matched 


def find_tilename_radec(ra,dec,dbh,schema='prod'):

    if ra<0:
        exit("ERROR: Please provide RA>0 and RA<360")

    QUERY_TILENAME_RADEC = {}
    # We match old table COADDTILE with COADDTILE_GEOM to match CROSSRA0 column
    QUERY_TILENAME_RADEC['des_admin'] = """
    select c.TILENAME from des_admin.COADDTILE c, prod.COADDTILE_GEOM g 
           where c.TILENAME=g.TILENAME AND 
                 (g.CROSSRA0='N' AND ({RA} BETWEEN URALL and URAUR) AND ({DEC} BETWEEN UDECLL and UDECUR)) OR
                 (g.CROSSRA0='Y' AND ({RA180} BETWEEN URALL-360 and URAUR-360) AND ({DEC} BETWEEN UDECLL and UDECUR)) 
    """
    # Special case for CROSSRA0
    QUERY_TILENAME_RADEC['prod']= """
    select TILENAME from prod.COADDTILE_GEOM 
           where (CROSSRA0='N' AND ({RA} BETWEEN RACMIN and RACMAX) AND ({DEC} BETWEEN DECCMIN and DECCMAX)) OR
                 (CROSSRA0='Y' AND ({RA180} BETWEEN RACMIN-360 and RACMAX) AND ({DEC} BETWEEN DECCMIN and DECCMAX))
    """

    # Case for DR/DR1
    QUERY_TILENAME_RADEC['dr1']= """
    select TILENAME from des_admin.DR1_TILE_INFO 
           where (CROSSRA0='N' AND ({RA} BETWEEN RACMIN and RACMAX) AND ({DEC} BETWEEN DECCMIN and DECCMAX)) OR
                 (CROSSRA0='Y' AND ({RA180} BETWEEN RACMIN-360 and RACMAX) AND ({DEC} BETWEEN DECCMIN and DECCMAX))
    """

    if ra > 180:
        ra180 = 360-ra
    else:
        ra180 = ra
    tilenames_dict = despyastro.query2dict_of_columns(QUERY_TILENAME_RADEC[schema].format(RA=ra,DEC=dec,RA180=ra180),dbh,array=False)
    
    if len(tilenames_dict)<1:
        SOUT.write("# WARNING: No tile found at ra:%s, dec:%s\n" % (ra,dec))
        return False
    else:
        return tilenames_dict['TILENAME'][0]
    return

def find_tilenames_radec(ra,dec,dbh,schema='prod'):

    """
    Find the tilename for each ra,dec and bundle them as dictionaries per tilename
    """

    indices = {}
    tilenames = []
    tilenames_matched = []
    for k in range(len(ra)):

        tilename = find_tilename_radec(ra[k],dec[k],dbh,schema=schema)
        tilenames_matched.append(tilename)

        # Write out the results
        if not tilename: # No tilename found
            # Here we could do something to store the failed (ra,dec) pairs
            continue

        # Store unique values and initialize list of indices grouped by tilename
        if tilename not in tilenames:
            indices[tilename]  = []
            tilenames.append(tilename)

        indices[tilename].append(k)

    return tilenames, indices, tilenames_matched


def get_coaddfiles_tilename_bytag(tilename,dbh,tag,bands='all',schema='prod'):


    QUERY_COADDFILES_BANDS = {}
    QUERY_COADDFILES_ALL = {}

    QUERY_COADDFILES_BANDS['des_admin'] = """
    select distinct f.path, TILENAME, BAND from des_admin.COADD c, des_admin.filepath_desar f
             where
             c.BAND in ({BANDS}) and
             c.TILENAME='{TILENAME}' and
             f.ID=c.ID and
             c.RUN in (select RUN from des_admin.RUNTAG where TAG='{TAG}')"""

    QUERY_COADDFILES_ALL['des_admin'] = """
    select distinct f.path, TILENAME, BAND from des_admin.COADD c, des_admin.filepath_desar f
             where
             c.BAND IS NOT NULL and
             c.TILENAME='{TILENAME}' and
             f.ID=c.ID and
             c.RUN in (select RUN from des_admin.RUNTAG where TAG='{TAG}')"""


    QUERY_COADDFILES_BANDS['prod'] = """
    select c.FILENAME, c.TILENAME, c.BAND, f.PATH, f.COMPRESSION from prod.COADD c, prod.PROCTAG, prod.FILE_ARCHIVE_INFO f
            where
              c.BAND in ({BANDS}) and
              prod.PROCTAG.TAG='{TAG}' and
              c.PFW_ATTEMPT_ID=PROCTAG.PFW_ATTEMPT_ID and
              f.FILENAME=c.FILENAME and
              c.TILENAME='{TILENAME}'"""

    QUERY_COADDFILES_ALL['prod'] = """
    select c.FILENAME, c.TILENAME, c.BAND, f.PATH, f.COMPRESSION from prod.COADD c, prod.PROCTAG, prod.FILE_ARCHIVE_INFO f
            where prod.PROCTAG.TAG='{TAG}' and
              c.PFW_ATTEMPT_ID=PROCTAG.PFW_ATTEMPT_ID and
              f.FILENAME=c.FILENAME and
              c.TILENAME='{TILENAME}'"""

    QUERY_COADDFILES_ALL['dr1'] = """
    select FITS_IMAGE_DET,
           FITS_IMAGE_G,
           FITS_IMAGE_R,
           FITS_IMAGE_I,
           FITS_IMAGE_Z,
           FITS_IMAGE_Y
       from des_admin.DR1_TILE_INFO
       where TILENAME='{TILENAME}'
       """

    QUERY_COADDFILES_BANDS['dr1'] = """
    select {FITS_IMAGES}
       from des_admin.DR1_TILE_INFO
       where TILENAME='{TILENAME}'
       """

    if bands == 'all':
        print QUERY_COADDFILES_ALL[schema].format(TILENAME=tilename,TAG=tag)
        rec = despyastro.query2rec(QUERY_COADDFILES_ALL[schema].format(TILENAME=tilename,TAG=tag),dbh)
    else:
        sbands = "'" + "','".join(bands) + "'" # trick to format
        fits_images = ", ".join(["FITS_IMAGE_{}".format(band.upper()) for band in bands]) 
        print QUERY_COADDFILES_BANDS[schema].format(TILENAME=tilename,TAG=tag,BANDS=sbands,FITS_IMAGES=fits_images)
        rec = despyastro.query2rec(QUERY_COADDFILES_BANDS[schema].format(TILENAME=tilename,TAG=tag,BANDS=sbands,FITS_IMAGES=fits_images),dbh)

    # Return a record array with the query
    return rec 

def get_avail_bands_dr1(filenames):
    ''' Get the available bands for the DR1 query '''

    avail_bands = []
    for name in filenames.dtype.names:
        if filenames[name] is not None:
            avail_bands.append(name[-1].lower())

    return avail_bands

def fix_compression(rec):

    # Here we fix 'COMPRESSION from None --> '' if present
    if rec is False:
        pass
    elif 'COMPRESSION' in rec.dtype.names:
        compression = [ '' if c is None else c for c in rec['COMPRESSION'] ]
        rec['COMPRESSION'] = numpy.array(compression)
    return rec

def get_coaddfiles_tilename_byid(tilename,id,dbh,coaddtable,bands='all'):

    """
    *** NOT USED AT THIS POINT ***
    An alternate way of searching for the tilenames' corresponding
    files by the COADD_OBJECT_ID in a given COADD_TABLE
    """

    QUERY_COADDFILES_ID_ALL = """
    select distinct f.path, TILENAME, BAND from des_admin.COADD c, des_admin.filepath_desar f
            where
            c.BAND IS NOT NULL and 
            and f.ID=c.ID and
            c.TILENAME='{TILENAME}'
            and c.RUN in (select run from des_admin.{COADDTABLE} where COADD_OBJECTS_ID={ID})
            """
    QUERY_COADDFILES_ID_BANDS = """
    select distinct f.path, TILENAME, BAND from des_admin.COADD c, des_admin.filepath_desar f
            where
            c.BAND in ({BANDS}) and
            and f.ID=c.ID and
            c.TILENAME='{TILENAME}'
            and c.RUN in (select run from des_admin.{COADDTABLE} where COADD_OBJECTS_ID={ID})
            """
    if bands == 'all':
        rec = despyastro.query2rec(QUERY_COADDFILES_ID_ALL.format(TILENAME=tilename,ID=id,COADDTABLE=coaddtable),dbh)
    else:
        sbands = "'" + "','".join(bands) + "'" # trick to format
        rec = despyastro.query2rec(QUERY_COADDFILES_ID_BANDS.format(TILENAME=tilename,ID=id,COADDTABLE=coaddtable,BANDS=sbands),dbh)
        
    # Return a record array with the query
    return rec 

def get_tilenames_in_tag(dbh,tag):

    Q_TILENAMES_TAG = """
    select distinct TILENAME from des_admin.COADD c, des_admin.filepath_desar f
           where
             f.ID=c.ID and
             c.RUN in (select RUN from des_admin.RUNTAG where TAG='{TAG}')"""
    return despyastro.query2rec(Q_TILENAMES_TAG.format(TAG=tag),dbh)
    