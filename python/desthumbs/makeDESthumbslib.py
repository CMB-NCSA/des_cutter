import argparse
import pandas
# import time
import sys
# from pyaml_env import parse_config
import multiprocessing as mp
import desthumbs
import desthumbs.fitsfinder as fitsfinder
# import desthumbs.thumbslib as thumbslib
import os
# import psutil
# import copy
import oracledb

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
    parser.add_argument("--tag", type=str, action="store", default = 'Y1A1_COADD',
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
    parser.add_argument("--db_section", type=str, action='store', default='db-dessci',
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
        sout.write("# \t--%-10s\t%s\n" % (key, vars(args)[key]))
    return args



def run(args):

    # The write log handle
    sout = args.sout
    desthumbs.fitsfinder.SOUT = args.sout
    desthumbs.thumbslib.SOUT = args.sout
    schema = 'des_admin'

    # Read in CSV file with pandas
    df = pandas.read_csv(args.inputList)

     #Get the arrays with ra, dec, xsize, ysize
    (xsize, ysize) = fitsfinder.check_xysize(df, xsize=args.xsize, ysize=args.ysize)
    args.ra = df.RA.values.tolist()
    args.dec = df.DEC.values.tolist()
    if 'OBJID' in df.keys():
        args.objID = df.OBJID.values.tolist()
    else:
        args.objID = None

    # Connect, get query and run query
    config_file = os.path.join(os.environ['HOME'], 'dbconfig.ini')
    # Get the connection credentials and information
    creds = fitsfinder.load_db_config(config_file, args.db_section)
    dbh = oracledb.connect(user=creds['user'],
                           password=creds['passwd'],
                           dsn=creds['dsn'])

    print("Connected")
    archive_root = desthumbs.get_archive_root(dbh, schema=schema, verb=True)
    print(archive_root)
    exit()


    tilenames, indices, tilenames_matched = find_tilenames_radec(ra, dec, dbh, schema=schema)
    exit()
