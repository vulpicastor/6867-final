#!/usr/bin/python3

from astropy import table
from astropy.io import fits, ascii
import logging
from os import path
import random
import sys
import traceback

import injection

BOUND_TOL = 6. / 24  # day; this is equivalent to 12 long cadences.
RANDOMIZE_BOUND = True
BOUND_TOL_UPPER = 1.  # day
BOUND_TOL_LOWER = 3./24  # day
NEG_UPPER = 1.  # day; upper bound for time span of negative samples.
NEG_LOWER = 3. / 24  # day; lower bound for time span of negative samples.

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
    if RANDOMIZE_BOUND:
        # For LSTM, padding should be randomized in order to prevent overfitting based on
        # when the transit starts in the light curve.
        half_width = dur / 2.
        start = mid_transit - half_width - random.range(BOUND_TOL_LOWER, BOUND_TOL_UPPER)
        stop = mid_transit + half_width + random.range(BOUND_TOL_LOWER, BOUND_TOL_UPPER)
    else:
        half_width = dur / 2. + BOUND_TOL  # Include an extra hour (12 long cadences)
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

def strip_rows_negative(fits_table, t_start, t_stop):
    header, data = fits_table.header, fits_table.data
    # t_start, t_stop = header["TSTART"], header["TSTOP"]
    dur = random.uniform(NEG_LOWER, NEG_UPPER)  # Choose a duration for the negative sample
    # Randomly find a start and stop time that does not overlap with interval
    for i in range(100):
        i_start = random.choice(data["TIME"])
        i_stop = i_start + dur
        if not (((t_start <= i_start) and (i_start <= t_stop)) or ((t_start <= i_stop) and (i_stop <= t_stop))):
            break
    else:
        logging.error("Cannot find a negative sample!")
        return None
    rows = []
    for r in data:
        if i_start <= r["TIME"] and r["TIME"] <= i_stop:
            rows.append(r)
    return rows

def strip_cols(fits_rows, metadata, mid_transit, is_eb, dur):
    data_rows = []
    start = mid_transit - dur / 2
    stop = mid_transit + dur / 2
    metadata["EB_injection"] = is_eb
    for r in fits_rows:
        # in_transit is 0 if outside transit and 1 if inside transit.
        in_transit = int((start <= r["TIME"]) and (r["TIME"] <= stop))
        eb_injection = int(is_eb and in_transit)
        data_rows.append((r["TIME"], r["SAP_FLUX"], in_transit, eb_injection))
    t = table.Table(rows=data_rows, names=("TIME", "SAP_FLUX", "IN_TRANSIT", "EB_injection"),
        dtype=("f8", "f8", "i4", "i4"), meta=metadata)
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
        # Also generate the negative samples
        neg_rows = strip_rows_negative(hdulist[1], start, stop)
        # Extract information of whether an eclipsing binary is being simulated
        is_eb = injected_row["EB_injection"]
        # Only preserve the TIME and SAP_FLUX columns of the light curve.
        # Also add tag for whether the mid transit point has passed.
        t = strip_cols(rows, dictify(hdulist[1].header), mid_transit, is_eb, dur)
        if neg_rows is None or is_eb:
            neg_t = None
        else:
            neg_t = strip_cols(neg_rows, dictify(hdulist[1].header), mid_transit, 0, dur)
    return t, neg_t

def main():
    logging.warning("Making sure that warning shots are fired")
    # Load the injected transits data table.
    injected = injection.parse_injected_table("/mnt/data/meta/kplr_dr25_inj3_plti.txt")
    index = injection.index_injected_table(injected)
    # Iterate through all filenames given as command line arguments.
    for i in sys.argv[1:]:
        logging.info("Processing %s", i)
        try:
            filename = path.abspath(i)
            # Example of an okay light curve:
            # "/mnt/data/INJ1/kplr011183555-2011271113734_INJECTED-inj1_llc.fits.gz"
            t, neg_t = gather_data(filename, injected, index)
            root = path.basename(filename).split(".")[0]
            ascii.write(t, root + "_quicklook.ecsv", format='ecsv')
            if neg_t is not None:
                ascii.write(neg_t, root + "_quicklook_negative.ecsv", format='ecsv')
        except Exception as e:
            logging.error("Error while processing file %s: %s", i, e)
            logging.error("Traceback: %s", traceback.format_exc())

if __name__ == "__main__":
    logging.basicConfig(filename="output.log", level=logging.INFO)
    main()
