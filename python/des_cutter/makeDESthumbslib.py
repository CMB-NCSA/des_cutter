#!/usr/bin/env python
import os
import sys
import time
import multiprocessing as mp
import argparse
import pandas
import duckdb
import des_cutter
import logging
import yaml
import datetime
import des_cutter.regexlib as regexlib
from des_cutter import fitsfinder
from des_cutter import thumbslib


def cmdline():

    # 1. Make a proto-parse use to read in the default yaml configuration
    # file, Turn off help, so we print all options in response to -h
    conf_parser = argparse.ArgumentParser(add_help=False)
    conf_parser.add_argument("-c", "--configfile", help="HeaderService config file")
    args, remaining_argv = conf_parser.parse_known_args()
    # If we have -c or --config, then we proceed to read it
    if args.configfile:
        conf_defaults = yaml.safe_load(open(args.configfile))
    else:
        conf_defaults = {}

    # 2. This is the main parser
    parser = argparse.ArgumentParser(description="Retrieves FITS fits within DES/DECADE and creates thumbnails for a list \
                                                  of sky positions",
                                     # Inherit options from config_parser
                                     parents=[conf_parser])
    # The positional arguments
    parser.add_argument("inputList", help="Input CSV file with positions (RA,DEC)"
                        "and optional (XSIZE,YSIZE) in arcmins")
    # Coadd and finalcut
    parser.add_argument("--coadd", action="store_true", default=False,
                        help="Run coadd cutouts")
    parser.add_argument("--finalcut", action="store_true", default=False,
                        help="Run finalcut cutouts")
    # The optional arguments for image retrieval
    parser.add_argument("--xsize", type=float, action="store", default=None,
                        help="Length of x-side in arcmins of image [default = 1]")
    parser.add_argument("--ysize", type=float, action="store", default=None,
                        help="Length of y-side of in arcmins image [default = 1]")
    parser.add_argument("--dbname", action='store', default="/home/felipe/dblib/des_metadata.duckdb",
                        help="Name of the duckdb database file")
    parser.add_argument("--bands", type=str, action='store', nargs='+', default='all',
                        help="Bands used for images. Can either be 'all' "
                        "(uses all bands, and is the default), or a list of individual bands")
    parser.add_argument("--prefix", type=str, action='store', default=thumbslib.PREFIX,
                        help=f"Prefix for thumbnail filenames [default={thumbslib.PREFIX}]")
    parser.add_argument("--tag", type=str, action='store', default='Y6A2',
                        help="Table TAG to use [default='Y6A2'")
    parser.add_argument("--date_start", type=str, action='store', default=None,
                        help="The START date to search for files formatted [YYYY-MM-DD]")
    parser.add_argument("--date_end", type=str, action='store', default=None,
                        help="The END date to search for files formatted [YYYY-MM-DD]")
    parser.add_argument("--colorset", type=str, action='store', nargs='+', default=['i', 'r', 'g'],
                        help="Color Set to use for creation of color image [default=i r g]")
    # Use multiprocessing
    parser.add_argument("--np", action="store", default=1, type=int,
                        help="Run using multi-process, 0=automatic, 1=single-process [default]")
    parser.add_argument("--outdir", type=str, action='store', default=os.getcwd(),
                        help="Output directory location [default='./']")

    # Logging options (loglevel/log_format/log_format_date)
    if 'LOG_LEVEL' in os.environ:
        default_log_level = os.environ['LOG_LEVEL']
    else:
        default_log_level = 'INFO'
    default_log_format = '[%(asctime)s.%(msecs)03d][%(levelname)s][%(name)s][%(funcName)s] %(message)s'
    default_log_format_date = '%Y-%m-%d %H:%M:%S'
    default_log = "des_cutter_{}.log".format(datetime.datetime.now().strftime("%Y-%m-%d-%H%M%S"))
    parser.add_argument("--loglevel", action="store", default=default_log_level, type=str.upper,
                        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
                        help="Logging Level [DEBUG/INFO/WARNING/ERROR/CRITICAL]")
    parser.add_argument("--log_format", action="store", type=str, default=default_log_format,
                        help="Format for logging")
    parser.add_argument("--log_format_date", action="store", type=str, default=default_log_format_date,
                        help="Format for date section of logging")
    parser.add_argument("--logfile", action="store", type=str, default=None,
                        help="Logfile where to send logging information")
    # args = parser.parse_args()
    # Set the defaults of argparse using the values in the yaml config file
    parser.set_defaults(**conf_defaults)
    args = parser.parse_args(args=remaining_argv)
    # Update  variables in config with actual values
    args.__dict__ = regexlib.replace_variables_in_dict(args.__dict__)
    args.loglevel = getattr(logging, args.loglevel)
    # Make sure that both date_start/end are defined or both are None
    if args.date_start is None and args.date_end is None:
        pass
    elif isinstance(args.date_start, str) and isinstance(args.date_end, str):
        pass
    elif isinstance(args.date_start, datetime.date) and isinstance(args.date_end, datetime.date):
        pass
    else:
        raise ValueError('Both --date_start and --date_end must be defined')

    # Set default logfile in case is not defined
    if args.logfile is None:
        args.logfile = os.path.join(args.outdir, default_log)

    # Logic for MP and NP
    # Get the number of processors to use
    args.NP = thumbslib.get_NP(args.np)
    if args.NP > 1 or args.MP:
        args.MP = True
    else:
        args.MP = False

    return args


def check_inputlist(inputList):

    # Read in CSV file with pandas
    df = pandas.read_csv(inputList)
    ra = df.RA.values  # if you only want the values otherwise use df.RA
    dec = df.DEC.values
    req_cols = ['RA', 'DEC']
    # Check columns for consistency
    fitsfinder.check_columns(df.columns, req_cols)
    # Check the xsize and ysizes
    xsize, ysize = fitsfinder.check_xy(df)
    return ra, dec, xsize, ysize, df


def prepare(args):

    # Create logger
    logger = logging.getLogger(__name__)

    # Make sure that outdir exists
    if not os.path.exists(args.outdir):
        msg = f"Creating outdir: {args.outdir}"
        os.makedirs(args.outdir)
    else:
        msg = None

    thumbslib.create_logger(logger, level=args.loglevel, MP=args.MP,
                            logfile=args.logfile,
                            log_format=args.log_format,
                            log_format_date=args.log_format_date)
    logger.info(f"Received command call:\n{' '.join(sys.argv[0:-1])}")
    logger.info(f"Running spt3g_cutter:{des_cutter.__version__}")
    logger.info(f"Running with args: \n{args}")
    if msg:
        logger.info(msg)
    return


def run_finalcut(args):

    logger = logging.getLogger(__name__)
    t0 = time.time()

    # Check input list for consistency
    ra, dec, xsize, ysize, df = check_inputlist(args.inputList)

    # connect to the DuckDB database -- via filename
    dbh = duckdb.connect(args.dbname, read_only=True)

    # Get archive_root
    archive_root = fitsfinder.get_archive_root()

    # Get the number of processors to use
    if args.NP > 1:
        p = mp.Pool(processes=args.NP)
        logger.info(f"Will use {args.NP} processors for FINALCUT process")
        results = []

    # Find all of the tilenames, indices grouped per tile
    logger.info("Finding FINALCUT images for each input (ra, dec) position")
    # Get the list of finalcut filenames
    df_images = fitsfinder.find_finalcut_images(ra, dec, dbh,
                                                date_start=args.date_start,
                                                date_end=args.date_end,
                                                bands=args.bands)
    # Close the DB connection
    dbh.close()

    # Loop over all ra, dec postions and results.
    for i, (ra_val, dec_val) in enumerate(zip(ra, dec)):
        df = df_images[i]
        Nfiles = len(df)
        # Loop over the files for each position
        for k in range(Nfiles):
            filename = os.path.join(archive_root, df.FILE[k])
            counter = f"{k+1}/{Nfiles} files"
            ar = (filename, ra_val, dec_val)
            kw = {'xsize': xsize[i], 'ysize': ysize[i],
                  'units': 'arcmin', 'prefix': args.prefix,
                  'outdir': args.outdir, 'counter': counter}

            if args.NP > 1:
                # Get result to catch exceptions later, after close()
                s = p.apply_async(thumbslib.fitscutter, args=ar, kwds=kw)
                results.append(s)
            else:

                thumbslib.fitscutter(*ar, **kw)

    # Close/join mp processes
    if args.NP > 1:
        p.close()
        # Check for exceptions
        for r in results:
            r.get()
        p.join()
        p.terminate()
        del p

    logger.info(f"Grand Total FINALCUT time:{thumbslib.elapsed_time(t0)}")


def run(args):

    t0 = time.time()
    # Get the logger
    logger = logging.getLogger(__name__)
    prepare(args)
    if args.coadd:
        run_coadd(args)
    if args.finalcut:
        run_finalcut(args)
    logger.info(f"FINAL time:{thumbslib.elapsed_time(t0)}")


def run_coadd(args):

    # Get the logger
    logger = logging.getLogger(__name__)

    # Check input list for consistency
    ra, dec, xsize, ysize, df = check_inputlist(args.inputList)

    # connect to the DuckDB database -- via filename
    dbh = duckdb.connect(args.dbname, read_only=True)

    # Get archive_root
    archive_root = fitsfinder.get_archive_root()

    # Find all of the tilenames, indices grouped per tile
    logger.info("Finding tilename for each input position")
    tilenames, indices, tilenames_matched = fitsfinder.find_tilenames_radec(ra, dec, dbh, tag=args.tag)

    # Add them back to pandas dataframe and write a file
    df['TILENAME'] = tilenames_matched
    # Get the thumbname base names and the them the pandas dataframe too
    df['THUMBNAME'] = thumbslib.get_base_names(tilenames_matched, ra, dec, prefix=args.prefix)
    matched_list = os.path.join(args.outdir, 'matched_'+os.path.basename(args.inputList))
    df.to_csv(matched_list, index=False)
    logger.info(f"Wrote matched tilenames list to: {matched_list}")

    # Store the files used
    files_used = os.path.join(args.outdir, 'files_used_'+os.path.basename(args.inputList))
    f_used = open(files_used, 'w')

    # Loop over all of the tilenames
    t0 = time.time()
    Ntile = 0
    for tilename in tilenames:
        t1 = time.time()
        Ntile = Ntile + 1
        logger.info("----------------------------------------------------")
        logger.info(f"Doing: {tilename} [{Ntile}/{len(tilenames)}]")
        logger.info("----------------------------------------------------")

        # 1. Get all of the filenames for a given tilename
        filenames = fitsfinder.get_coaddfiles_tilename(tilename, dbh, tag=args.tag, bands=args.bands)

        if filenames is False:
            logger.info(f"Skipping: {tilename} -- not in TABLE")
            continue

        indx = indices[tilename]
        avail_bands = filenames.BAND
        if args.NP > 1:
            NP = min(len(avail_bands), args.NP)
        else:
            NP = 1
        logger.info(f"Will use {NP} processors for COADD processing")

        # 2. Loop over all of the filename -- We could use multi-processing
        p = {}
        n_filenames = len(avail_bands)
        for k in range(n_filenames):

            # Rebuild the full filename with COMPRESSION if present
            if 'COMPRESSION' in filenames.dtype.names:
                filename = os.path.join(archive_root, filenames.PATH[k], filenames.FILENAME[k])+filenames.COMPRESSION[k]
            else:
                filename = os.path.join(archive_root, filenames.PATH[k])

            # Write them to a file
            f_used.write(filename+"\n")
            counter = f"{k+1}/{n_filenames} files"
            ar = (filename, ra[indx], dec[indx])
            kw = {'xsize': xsize[indx], 'ysize': ysize[indx],
                  'units': 'arcmin', 'prefix': args.prefix, 'outdir': args.outdir,
                  'tilename': tilename, 'counter': counter}
            logger.info(f"Cutting: {filename}")
            if NP > 1:
                p[filename] = mp.Process(target=thumbslib.fitscutter, args=ar, kwargs=kw)
                p[filename].start()
            else:
                thumbslib.fitscutter(*ar, **kw)

        # Make sure all process are closed before proceeding
        if NP > 1:
            for filename, value in p.items():
                value.join()

        # 3. Create color images using stiff for each ra,dec and loop over (ra,dec)
        for k in range(len(ra[indx])):
            des_cutter.color_radec(ra[indx][k], dec[indx][k], avail_bands,
                                   prefix=args.prefix,
                                   colorset=args.colorset,
                                   outdir=args.outdir,
                                   stiff_parameters={'NTHREADS': NP})

        logger.info(f"Time {tilename}: {thumbslib.elapsed_time(t1)}")

    # Close the DB connection
    dbh.close()
    f_used.close()
    logger.info(f"Grand Total time:{thumbslib.elapsed_time(t0)}")
    return
