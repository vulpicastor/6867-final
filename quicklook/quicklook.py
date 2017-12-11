#!/usr/bin/python3

from astropy import table
from astropy.io import fits, ascii
import logging
from os import path
import sys
import traceback

import injection

BOUND_TOL = 6. / 24  # day; this is equivalent to 12 long cadences.

def dictify(fits_header):
    """-> dictionary representation of the FITS header.

    This is necessary because undefined cards can't be serialized and need to
    be turned in to Nones.
    """
    out = {}
    for k, v in fits_header.items():
        if type(v) is not fits.card.Undefined:
            out[k] = v
        else:
            out[k] = None
    return out

def find_time_range(fits_table_header, injected_row):
    epoch = injected_row["i_epoch"]  # mid-transit reference epoch.
    period = injected_row["i_period"]  # transit period (i.e. how often it repeats).
    dur = injected_row["i_dur"] / 24.  # transit duration in days.
    # Find the start and stop times of the light curve.
    t_start, t_stop = fits_table_header["TSTART"], fits_table_header["TSTOP"]
    # Find how many periods since epoch is the transit in the light curve.
    num_period = (t_start - epoch) // period + 1  # VERY IMPORTANT OFF BY ONE.
    # Calculate the start and stop times of the transit.
    mid_transit = epoch + period * num_period  # Time of mid-transit
    half_width = dur / 2 + BOUND_TOL  # Include an extra hour (12 long cadences)
    start = mid_transit - half_width
    stop = mid_transit + half_width
    logging.info("The start and stop period for {} is {}, {}".format(fits_table_header["OBJECT"], start, stop))
    return start, stop, mid_transit, dur

def strip_rows(fits_table, t_start, t_stop):
    rows = []
    header, data = fits_table.header, fits_table.data
    # Make sure that the given start and stop times are actually within bounds
    # of the light curve.
    if header["TSTART"] >= t_start or header["TSTOP"] <= t_stop:  # Assumes that table is sorted by time.
        logging.warning("t_start or t_stop outside of light curve bounds!")
    for r in data:
        if t_start <= r["TIME"] and r["TIME"] <= t_stop:
            rows.append(r)
    return rows

def strip_cols(fits_rows, metadata, mid_transit, dur):
    data_rows = []
    start = mid_transit - dur / 2
    stop = mid_transit + dur / 2
    for r in fits_rows:
        # in_transit is 0 if outside transit and 1 if inside transit.
        in_transit = int((start <= r["TIME"]) and (r["TIME"] <= stop))
        data_rows.append((r["TIME"], r["SAP_FLUX"], in_transit))
    t = table.Table(rows=data_rows, names=("TIME", "SAP_FLUX", "IN_TRANSIT"),
        dtype=("f8", "f8", "i4"), meta=metadata)
    return t

def gather_data(filename, injected_table, injected_table_index):
    with fits.open(filename) as hdulist:
        # Extract KIC ID part of the file name.
        kic_id = injection.parse_injected_filename(filename)
        # Find corresponding row in the injected data table.
        injected_row = injected_table[injected_table_index[kic_id]]
        # Find when the transit starts and stops in the FITS file.
        start, stop, mid_transit, dur = find_time_range(hdulist[1].header, injected_row)
        # Strip out the rows that are actually transts.
        rows = strip_rows(hdulist[1], start, stop)
        # Only preserve the TIME and SAP_FLUX columns of the light curve.
        # Also add tag for whether the mid transit point has passed.
        t = strip_cols(rows, dictify(hdulist[1].header), mid_transit, dur)
    return t

def main():
    logging.warning("Making sure that warning shots are fired")
    # Load the injected transits data table.
    injected = injection.parse_injected_table("/mnt/data/meta/kplr_dr25_inj1_plti.txt")
    index = injection.index_injected_table(injected)
    # Iterate through all filenames given as command line arguments.
    for i in sys.argv[1:]:
        logging.info("Processing %s", i)
        try:
            filename = path.abspath(i)
            # Example of an okay light curve:
            # "/mnt/data/INJ1/kplr011183555-2011271113734_INJECTED-inj1_llc.fits.gz"
            t = gather_data(filename, injected, index)
            root = path.basename(filename).split(".")[0]
            ascii.write(t, root + "_quicklook.ecsv", format='ecsv')
        except Exception as e:
            logging.error("Error while processing file %s: %s", i, e)
            logging.error("Traceback: %s", traceback.format_exc())

if __name__ == "__main__":
    logging.basicConfig(filename="output.log", level=logging.INFO)
    main()
