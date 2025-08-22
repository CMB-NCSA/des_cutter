#!/usr/bin/env python
import os
import sys
import time
import multiprocessing as mp
import argparse
import pandas
import duckdb
import des_cutter
from des_cutter import fitsfinder
from des_cutter import thumbslib

XSIZE_DEFAULT = 1.0
YSIZE_DEFAULT = 1.0


def cmdline():
    """Command line parser"""
    parser = argparse.ArgumentParser(description="Retrieves FITS fits within DES and creates thumbnails for a list \
                                                  of sky positions")
    # The positional arguments
    parser.add_argument("inputList", help="Input CSV file with positions (RA,DEC)"
                        "and optional (XSIZE,YSIZE) in arcmins")
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
    parser.add_argument("--prefix", type=str, action='store', default='DES',
                        help="Prefix for thumbnail filenames [default='DES']")
    parser.add_argument("--tag", type=str, action='store', default='Y6A2',
                        help="Table TAG to use [default='Y6A2'")
    parser.add_argument("--date_start", type=str, action='store', default=None,
                        help="The START date to search for files formatted [YYYY-MM-DD]")
    parser.add_argument("--date_end", type=str, action='store', default=None,
                        help="The END date to search for files formatted [YYYY-MM-DD]")
    parser.add_argument("--colorset", type=str, action='store', nargs='+', default=['i', 'r', 'g'],
                        help="Color Set to use for creation of color image [default=i r g]")
    # Use multiprocessing
    parser.add_argument("--MP", action='store_true', default=False,
                        help="Run in multiple core [default=False]")
    parser.add_argument("--np", action="store", default=1, type=int,
                        help="Run using multi-process, 0=automatic, 1=single-process [default]")
    parser.add_argument("--verb", action='store_true', default=False,
                        help="Turn on verbose mode [default=False]")
    parser.add_argument("--outdir", type=str, action='store', default=os.getcwd(),
                        help="Output directory location [default='./']")
    parser.add_argument("--logfile", type=str, action='store', default=None,
                        help="Output logfile")

    args = parser.parse_args()

    # Make sure that both date_start/end are defined or both are None
    if args.date_start is None and args.date_end is None:
        pass
    elif isinstance(args.date_start, str) and isinstance(args.date_end, str):
        pass
    else:
        raise ValueError('Both --date_start and --date_end must be defined')

    if args.logfile:
        sout = open(args.logfile, 'w', encoding="utf-8")
    else:
        sout = sys.stdout
    args.sout = sout
    sout.write("# Will run:\n")
    sout.write(f"# {parser.prog} \n")
    for key, value in vars(args).items():
        if key == "password":
            continue
        sout.write(f"# \t--{key:<10}\t{value}\n")
    return args


def run_finalcut(args):
    # The write log handle
    sout = args.sout
    des_cutter.fitsfinder.SOUT = args.sout
    des_cutter.thumbslib.SOUT = args.sout

    # Get the number of processors to use
    NP = thumbslib.get_NP(args.np)
    if NP > 1:
        MP = True
    else:
        MP = False

    if NP > 1:
        p = mp.Pool(processes=NP)
        print(f"Will use {NP} processors for process")
        manager = mp.Manager()
        cutout_dict = manager.dict()
        rejected_dict = manager.dict()
        results = []
    else:
        cutout_dict = None
        rejected_dict = None

    # Read in CSV file with pandas
    df = pandas.read_csv(args.inputList)
    ra = df.RA.values  # if you only want the values otherwise use df.RA
    dec = df.DEC.values
    nobj = len(ra)
    req_cols = ['RA', 'DEC']

    # Check columns for consistency
    fitsfinder.check_columns(df.columns, req_cols)

    # Check the xsize and ysizes
    xsize, ysize = fitsfinder.check_xysize(df, args, nobj)
    # connect to the DuckDB database -- via filename
    dbh = duckdb.connect(args.dbname, read_only=True)

    # Get archive_root
    archive_root = fitsfinder.get_archive_root(verb=True)

    # Make sure that outdir exists
    if not os.path.exists(args.outdir):
        if args.verb:
            sout.write(f"# Creating: {args.outdir}")
        os.makedirs(args.outdir)

    # Find all of the tilenames, indices grouped per tile
    if args.verb:
        sout.write("# Finding FINALCUT images for each input (ra, dec) position\n")
    # Get the list of finalcut filenames
    df_images = fitsfinder.find_finalcut_images(ra, dec, dbh,
                                                date_start=args.date_start,
                                                date_end=args.date_end,
                                                bands=args.bands)
    # Loop over all ra, dec postions and results.
    for i, (ra_val, dec_val) in enumerate(zip(ra, dec)):
        df = df_images[i]
        Nfiles = len(df)
        print(i, ra_val, dec_val, Nfiles)
        # Loop over the files for each position
        for k in range(Nfiles):
            filename = os.path.join(archive_root, df.FILE[k])
            # scamp_head = os.path.join(archive_root, df.SCAMP_HEAD[k])
            # counter = f"{k}/{Nfiles} files"
            ar = (filename, ra_val, dec_val)
            kw = {'xsize': xsize[i], 'ysize': ysize[i],
                  'units': 'arcmin', 'prefix': args.prefix,
                  'outdir': args.outdir, # 'counter': counter,
                  'verb': args.verb}

            if NP > 1:
                # Get result to catch exceptions later, after close()
                s = p.apply_async(thumbslib.fitscutter, args=ar, kwds=kw)
                results.append(s)
            else:

                thumbslib.fitscutter(*ar, **kw)
            #print(i, k, filename)

    # Close/join mp processes
    if NP > 1:
        p.close()
        # Check for exceptions
        for r in results:
            r.get()
        p.join()
        p.terminate()
        del p


def run(args):
    # The write log handle
    sout = args.sout
    des_cutter.fitsfinder.SOUT = args.sout
    des_cutter.thumbslib.SOUT = args.sout

    # Read in CSV file with pandas
    df = pandas.read_csv(args.inputList)
    ra = df.RA.values  # if you only want the values otherwise use df.RA
    dec = df.DEC.values
    nobj = len(ra)
    req_cols = ['RA', 'DEC']

    # Check columns for consistency
    fitsfinder.check_columns(df.columns, req_cols)

    # Check the xsize and ysizes
    xsize, ysize = fitsfinder.check_xysize(df, args, nobj)
    # connect to the DuckDB database -- via filename
    dbh = duckdb.connect(args.dbname, read_only=True)

    # Get archive_root
    archive_root = fitsfinder.get_archive_root(verb=True)

    # Make sure that outdir exists
    if not os.path.exists(args.outdir):
        if args.verb:
            sout.write(f"# Creating: {args.outdir}")
        os.makedirs(args.outdir)

    # Find all of the tilenames, indices grouped per tile
    if args.verb:
        sout.write("# Finding tilename for each input position\n")
    tilenames, indices, tilenames_matched = fitsfinder.find_tilenames_radec(ra, dec, dbh, tag=args.tag)

    # Add them back to pandas dataframe and write a file
    df['TILENAME'] = tilenames_matched
    # Get the thumbname base names and the them the pandas dataframe too
    df['THUMBNAME'] = thumbslib.get_base_names(tilenames_matched, ra, dec, prefix=args.prefix)
    matched_list = os.path.join(args.outdir, 'matched_'+os.path.basename(args.inputList))
    df.to_csv(matched_list, index=False)
    sout.write(f"# Wrote matched tilenames list to: {matched_list}\n")

    # Store the files used
    files_used = os.path.join(args.outdir, 'files_used_'+os.path.basename(args.inputList))
    f_used = open(files_used, 'w')

    # Loop over all of the tilenames
    t0 = time.time()
    Ntile = 0
    for tilename in tilenames:
        t1 = time.time()
        Ntile = Ntile + 1
        sout.write("# ----------------------------------------------------\n")
        sout.write(f"# Doing: {tilename} [{Ntile}/{len(tilenames)}]\n")
        sout.write("# ----------------------------------------------------\n")

        # 1. Get all of the filenames for a given tilename
        filenames = fitsfinder.get_coaddfiles_tilename(tilename, dbh, bands=args.bands)

        if filenames is False:
            sout.write(f"# Skipping: {tilename} -- not in TABLE \n")
            continue

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
                filename = os.path.join(archive_root, filenames.PATH[k])

            # Write them to a file
            f_used.write(filename+"\n")

            ar = (filename, ra[indx], dec[indx])
            kw = {'xsize': xsize[indx], 'ysize': ysize[indx],
                  'units': 'arcmin', 'prefix': args.prefix, 'outdir': args.outdir,
                  'tilename': tilename, 'verb': args.verb}
            if args.verb:
                sout.write(f"# Cutting: {filename}\n")
            if args.MP:
                NP = len(avail_bands)
                p[filename] = mp.Process(target=thumbslib.fitscutter, args=ar, kwargs=kw)
                p[filename].start()
            else:
                NP = 1
                thumbslib.fitscutter(*ar, **kw)

        # Make sure all process are closed before proceeding
        if args.MP:
            for filename, value in p.items():
                value.join()

        # 3. Create color images using stiff for each ra,dec and loop over (ra,dec)
        for k in range(len(ra[indx])):
            des_cutter.color_radec(ra[indx][k], dec[indx][k], avail_bands,
                                   prefix=args.prefix,
                                   colorset=args.colorset,
                                   outdir=args.outdir,
                                   verb=args.verb,
                                   stiff_parameters={'NTHREADS': NP})

        if args.verb:
            sout.write(f"# Time {tilename}: {thumbslib.elapsed_time(t1)}\n")

    f_used.close()
    sout.write(f"\n*** Grand Total time:{thumbslib.elapsed_time(t0)} ***\n")
    return
