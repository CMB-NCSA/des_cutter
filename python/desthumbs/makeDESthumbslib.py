#!/usr/bin/env python
import os
import sys
import pandas
import numpy
import time
import multiprocessing as mp

import desthumbs
import oracledb
import desthumbs.fitsfinder as fitsfinder
import argparse

XSIZE_default = 1.0
YSIZE_default = 1.0


def cmdline():
     parser = argparse.ArgumentParser(description="Retrieves FITS images within DES given the file and other parameters")

     # The positional arguments
     parser.add_argument("inputList", help="Input CSV file with positions (RA,DEC) and optional (XSIZE,YSIZE) in arcmins")

     # The optional arguments for image retrieval
     parser.add_argument("--xsize", type=float, action="store", default=None,
                         help="Length of x-side in arcmins of image [default = 1]")
     parser.add_argument("--ysize", type=float, action="store", default=None,
                         help="Length of y-side of in arcmins image [default = 1]")
     parser.add_argument("--tag", type=str, action="store", default = 'Y6A2',
                         help="Tag used for retrieving files [default=Y1A1_COADD]")
     parser.add_argument("--coaddtable", type=str, action="store", default=None ,
                         help="COADD table name to query if COADDS_ID are provided instead of RA,DEC in the input csv file")
     parser.add_argument("--bands", type=str, action='store', nargs = '+', default='all',
                         help="Bands used for images. Can either be 'all' (uses all bands, and is the default), or a list of individual bands")
     parser.add_argument("--prefix", type=str, action='store', default='DES',
                         help="Prefix for thumbnail filenames [default='DES']")
     parser.add_argument("--colorset", type=str, action='store', nargs = '+', default=['i','r','g'],
                         help="Color Set to use for creation of color image [default=i r g]")
     parser.add_argument("--MP", action='store_true', default=False,
                         help="Run in multiple core [default=False]")
     parser.add_argument("--verb", action='store_true', default=False,
                         help="Turn on verbose mode [default=False]")
     parser.add_argument("--outdir", type=str, action='store', default=os.getcwd(),
                         help="Output directory location [default='./']")
     parser.add_argument("--db_section", type=str, action='store',default='db-desoper',
                         help="Database section to connect to")
     parser.add_argument("--user", type=str, action='store',help="Username")
     parser.add_argument("--password", type=str, action='store', help="password")
     parser.add_argument("--logfile", type=str, action='store', default=None,
                         help="Output logfile")

     args = parser.parse_args()

     if args.logfile:
          sout = open(args.logfile,'w')
     else:
          sout = sys.stdout
     args.sout = sout
     sout.write("# Will run:\n")
     sout.write("# %s \n" % parser.prog)
     for key in vars(args):
         if key == 'password': continue
         sout.write("# \t--%-10s\t%s\n" % (key,vars(args)[key]))
     return args

def check_xysize(df,args,nobj):

    # Check if  xsize,ysize are set from command-line or read from csv file
    if args.xsize: xsize = numpy.array([args.xsize]*nobj)
    else:
        try: xsize = df.XSIZE.values
        except: xsize = numpy.array([XSIZE_default]*nobj)

    if args.ysize: ysize = numpy.array([args.ysize]*nobj)
    else:
        try: ysize = df.YSIZE.values
        except: ysize = numpy.array([YSIZE_default]*nobj)

    return xsize,ysize

def check_columns(cols,req_cols):

    """ Test that all required columns are present"""
    for c in req_cols:
         if c not in cols :
              raise TypeError('column %s in file' % c)
    return


def get_base_names(tilenames, ra, dec, prefix='DES'):
    names = []
    for k in range(len(ra)):
        if tilenames[k]:
            name = desthumbs.get_thumbBaseName(ra[k],dec[k],prefix=prefix)
        else:
            name = False
        names.append(name)
    return names

def run(args):

    # The write log handle
    sout = args.sout
    desthumbs.fitsfinder.SOUT = args.sout
    desthumbs.thumbslib.SOUT = args.sout

    # Read in CSV file with pandas
    df = pandas.read_csv(args.inputList)

    # Decide if we do search by RA,DEC or by COADD_ID
    # if 'COADD_OBJECTS_ID' in df.columns:
    #      searchbyID = True
    #      if not args.coaddtable:
    #           raise Exception("ERROR: Need to provide --coaddtable <tablename> when using COADD_OBJECTS_ID")
    #      coadd_id =  dec = df.COADD_OBJECTS_ID.values
    #      nobj = len(coadd_id)
    #      req_cols = ['COADD_OBJECTS_ID']
    # else:
    searchbyID = False
    ra  = df.RA.values  # if you only want the values otherwise use df.RA
    dec = df.DEC.values
    nobj = len(ra)
    req_cols = ['RA', 'DEC']

    # Check columns for consistency
    check_columns(df.columns, req_cols)

    # Check the xsize and ysizes
    xsize,ysize = check_xysize(df, args,nobj)

    # Get DB handle
    try:
    #   dbh = desdbi.DesDbi(section=args.db_section)
        dbh = 'db-dessci'
    except:
      if args.db_section == 'desoper' or args.db_section == 'db-desoper':
        host = 'desdb.ncsa.illinois.edu'
        port = '1521'
        name = 'desoper'
      elif args.db_section == 'oldsci' or args.db_section == 'db-oldsci':
        host = 'desdb-dr.ncsa.illinois.edu'
        port = '1521'
        name = 'desdr'

      kwargs = {'host': host, 'port': port, 'service_name': name}
    #   dsn = cx_Oracle.makedsn(**kwargs)
    #   dbh = cx_Oracle.connect(args.user, args.password, dsn=dsn)
    #This shoukd be config
    #  #move this into new makeDESthumbslib

    config_file = os.path.join(os.environ['HOME'], 'dbconfig.ini')
    # Get the connection credentials and information

    creds = fitsfinder.load_db_config(config_file, dbh)
    dbh = oracledb.connect(user=creds['user'],
                           password=creds['passwd'],
                           dsn=creds['dsn'])

    schema = 'des_admin'

    print("SCHEMA IS " + schema)
    # Get archive_root

    archive_root = fitsfinder.get_archive_root(dbh, schema=schema, verb=True)
    print(archive_root)

    # Make sure that outdir exists
    if not os.path.exists(args.outdir):
        if args.verb:
            sout.write("# Creating: %s\n" % args.outdir)
        os.makedirs(args.outdir)

    # Find all of the tilenames, indices grouped per tile
    if args.verb:
        sout.write("# Finding tilename for each input position\n")
    tilenames, indices, tilenames_matched = fitsfinder.find_tilenames_radec(ra,dec,dbh,schema=schema)

    # Add them back to pandas dataframe and write a file
    df['TILENAME'] = tilenames_matched
    # Get the thumbname base names and the them the pandas dataframe too
    df['THUMBNAME'] = get_base_names(tilenames_matched, ra, dec, prefix=args.prefix)
    matched_list = os.path.join(args.outdir, 'matched_'+os.path.basename(args.inputList))
    df.to_csv(matched_list, index=False)
    sout.write("# Wrote matched tilenames list to: %s\n" % matched_list)

    # Make sure that all found tilenames *are* in the tag (aka data exists for them)
    #tilenames_intag = desthumbs.get_tilenames_in_tag(dbh,args.tag)

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
        filenames = fitsfinder.get_coaddfiles_tilename_bytag(tilename, dbh, args.tag, bands=args.bands)
        print(filenames)

        if filenames is False:
            sout.write("# Skipping: %s -- not in TAG:%s \n" % (tilename, args.tag))
            continue
        # Fix compression for SV1/Y2A1/Y3A1 releases
        else:
            filenames = fitsfinder.fix_compression(filenames)

        indx = indices[tilename]

        # dr1 Schema?
        # if schema == 'dr1':
        #   avail_bands = desthumbs.get_avail_bands_dr1(filenames)
        # else:
        avail_bands = filenames.BAND

        # 2. Loop over all of the filename -- We could use multi-processing
        p = {}
        n_filenames = len(avail_bands)
        for k in range(n_filenames):

            # Rebuild the full filename with COMPRESSION if present
            if 'COMPRESSION' in filenames.dtype.names:
                filename = os.path.join(archive_root, filenames.PATH[k], filenames.FILENAME[k])+filenames.COMPRESSION[k]
            else:
                filename = os.path.join(archive_root, filenames.PATH[k])

            print(filename)
            ar = (filename, ra[indx], dec[indx])
            kw = {'xsize': xsize[indx], 'ysize': ysize[indx],
                  'units': 'arcmin', 'prefix': args.prefix, 'outdir': args.outdir,
                  'tilename': tilename, 'verb': args.verb}
            if args.verb:
                sout.write("# Cutting: %s\n" % filename)
            if args.MP:
                NP = len(avail_bands)
                p[filename] = mp.Process(target=desthumbs.fitscutter, args=ar, kwargs=kw)
                p[filename].start()
            else:
                NP = 1
                desthumbs.fitscutter(*ar, **kw)

        # Make sure all process are closed before proceeding
        if args.MP:
            for filename in p.keys(): p[filename].join()

        # 3. Create color images using stiff for each ra,dec and loop over (ra,dec)
        for k in range(len(ra[indx])):
            desthumbs.color_radec(ra[indx][k],dec[indx][k],avail_bands,
                                  prefix=args.prefix,
                                  colorset=args.colorset,
                                  outdir=args.outdir,
                                  verb=args.verb,
                                  stiff_parameters={'NTHREADS':NP})

        if args.verb: sout.write("# Time %s: %s\n" % (tilename,desthumbs.elapsed_time(t1)))

    sout.write("\n*** Grand Total time:%s ***\n" % desthumbs.elapsed_time(t0))
    return
