import argparse
import logging
import pandas
import time
import sys
# from pyaml_env import parse_config
import multiprocessing as mp
import desthumbsNEW
import desthumbsNEW.fitsfinder as fitsfinder
import desthumbsNEW.new_thumbslib as thumbslib
import os
import psutil
import copy

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

def run(args):
    logger = logging.getLogger(__name__)
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
    dbhandle = fitsfinder.connect_db(args.dbname)
    query = fitsfinder.get_query(args.tablename,
                                 bands=args.bands,
                                 filetypes=args.filetypes,
                                 date_start=args.date_start,
                                 date_end=args.date_end,
                                 yearly=args.yearly)
    logger.info(f"Running query: {query}")
    rec = fitsfinder.query2rec(query, dbhandle)

    cutout_names = {}
    rejected_names = {}
    lightcurve = {}

    # Get the number of processors to use
    NP = thumbslib.get_NP(args.np)
    if NP > 1:
        p = mp.Pool(processes=NP)
        logger.info(f"Will use {NP} processors for process")
        manager = mp.Manager()
        cutout_dict = manager.dict()
        rejected_dict = manager.dict()
        lightcurve_dict = manager.dict()
        results = []
    else:
        cutout_dict = None
        rejected_dict = None
        lightcurve_dict = None

    # Create the outdir if it does not exists
    if not os.path.exists(args.outdir):
        os.makedirs(args.outdir, mode=0o755, exist_ok=True)

    # Loop over all files
    args.files = rec['FILE'].tolist()
    Nfiles = len(args.files)
    logger.info(f"Found {Nfiles} files")
    k = 1
    t0 = time.time()
    for file in args.files:
        counter = f"{k}/{Nfiles} files"

        # Make a copy of objID if not None:
        if args.objID is None:
            objID = None
        else:
            objID = copy.deepcopy(args.objID)

        ar = (file, args.ra, args.dec, cutout_dict, rejected_dict, lightcurve_dict)
        kw = {'xsize': xsize, 'ysize': ysize, 'units': 'arcmin', 'objID': objID,
              'prefix': args.prefix, 'outdir': args.outdir, 'counter': counter,
              'get_lightcurve': args.get_lightcurve,
              'get_uniform_coverage': args.get_uniform_coverage,
              'nofits': args.nofits,
              'stage': args.stage,
              'stage_prefix': args.stage_prefix}

        if NP > 1:
            # Get result to catch exceptions later, after close()
            s = p.apply_async(thumbslib.fitscutter, args=ar, kwds=kw)
            results.append(s)
        else:
            names, pos, lc = thumbslib.fitscutter(*ar, **kw)
            cutout_names.update(names)
            rejected_names.update(pos)
            lightcurve.update(lc)
        k += 1

    if NP > 1:
        p.close()
        # Check for exceptions
        for r in results:
            r.get()
        p.join()

        # Update with returned dictionary, we need to make them real
        # dictionaries, instead DictProxy objects returned from multiprocessing
        logger.info("Updating returned dictionaries")
        cutout_names = cutout_dict.copy()
        rejected_names = rejected_dict.copy()
        lightcurve = lightcurve_dict.copy()
        p.terminate()
        del p

    # Time it took to just cut
    logger.info(f"Total cutting time: {thumbslib.elapsed_time(t0)}")

    # Store the dict with all of the cutout names and rejects
    args.cutout_names = cutout_names
    args.rejected_names = rejected_names

    args = thumbslib.capture_job_metadata(args)

    # Report total memory usage
    logger.info(f"Memory: {psutil.Process(os.getpid()).memory_info().rss / 1024 ** 3} Gb")
    process = psutil.Process(os.getpid())
    logger.info(f"Memory percent: {process.memory_percent()} %")

    # # Clean up
    if NP > 1:
        logger.info("Deleting variables -- probably futile")
        del manager
        del cutout_names
        del cutout_dict
        del rejected_names
        del rejected_dict
        del lightcurve_dict

    if args.get_lightcurve:

        # Get the observations dictionary
        args.obs_dict = thumbslib.get_obs_dictionary(lightcurve)
        logger.info(f"Size of lightcurve: {sys.getsizeof(lightcurve)/1024/1024}")
        logger.info(f"Size of args.obs_dict: {sys.getsizeof(args.obs_dict)/1024/1024}")

        # Create new pool
        # NP = 1  # Remove this line once we have a machine with more memory
        NP = len(args.bands)
        logger.info(f"Creating pool with: {NP} processes for repack lightcurve")
        if NP > 1:
            p = mp.Pool(processes=NP)
            results = []

        for BAND in args.bands:
            for FILETYPE in args.filetypes:
                logger.info(f"Memory: {psutil.Process(os.getpid()).memory_info().rss / 1024 ** 3} Gb")
                ar = (lightcurve, BAND, FILETYPE, args)
                logger.info(f"Memory percent: {process.memory_percent()} %")
                if NP > 1:
                    s = p.apply_async(thumbslib.repack_lightcurve_band_filetype, args=ar)
                    results.append(s)
                else:
                    thumbslib.repack_lightcurve_band_filetype(*ar)

        if NP > 1:
            p.close()
            # Check for exceptions
            for r in results:
                r.get()
            p.join()
            p.terminate()
            del p

    # Write the manifest file
    thumbslib.write_manifest(args)
    logger.info(f"Grand Total time: {thumbslib.elapsed_time(t0)}")


    
