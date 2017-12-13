#!/usr/bin/python3

from astropy import table
from astropy.io import fits, ascii
import logging
from os import path
import random
import sys
import traceback

import injection

# Whether to randomize padding. if True, will be randomly selected between
# BOUND_TOL_LOWER and BOUND_TOL_UPPER. If False, will use fixed BOUND_TOL.
# True for LSTM, False for SVM.
RANDOMIZE_BOUND = True
BOUND_TOL = 6. / 24  # day; this is equivalent to 12 long cadences.
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

def find_time_range(table_start, table_stop, epoch, period, dur, name="Unknown"):
    # Find how many periods since epoch is the transit in the light curve.
    num_period = (table_start - epoch) // period + 1  # VERY IMPORTANT OFF BY ONE.
    # Calculate the start and stop times of the transit.
    mid_transit = epoch + period * num_period  # Time of mid-transit
    half_width = dur / 2.
    transit_start = mid_transit - half_width
    transit_stop = mid_transit + half_width
    # literally the transit don't exist in the data.
    if table_start >= transit_stop or table_stop <= transit_start:
        logging.error("Light curve range contains no transit for %s!", name)
        return None, None, None, None
    # Sanity checking.
    if RANDOMIZE_BOUND:
        # For LSTM, padding should be randomized in order to prevent overfitting based on
        # when the transit starts in the light curve.
        start = transit_start - random.uniform(BOUND_TOL_LOWER, BOUND_TOL_UPPER)
        stop = transit_stop + random.uniform(BOUND_TOL_LOWER, BOUND_TOL_UPPER)
    else:
        # Constant padding otherwise.
        start = transit_start - BOUND_TOL
        stop = transit_stop - BOUND_TOL
    # Check that transit is not clipped by data boundary
    if table_start >= start or table_stop <= stop:
        logging.error("Transit partially outside light curve range for %s!", name)
        return None, None, None, None
    logging.info("The start and stop period for {} is {}, {}".format(name, start, stop))
    return start, stop, mid_transit, dur

def strip_rows(fits_table, t_start, t_stop):
    rows = []
    header, data = fits_table.header, fits_table.data
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
        injected_i = injected_table_index[kic_id]
        # Extract row metadata
        epoch = injected_table["i_epoch"][injected_i]
        period = injected_table["i_period"][injected_i]
        dur = injected_table["i_dur"][injected_i] / 24.  # hour
        table_start = hdulist[1].header["TSTART"]
        table_stop = hdulist[1].header["TSTOP"]
        # Find when the transit starts and stops in the FITS file.
        start, stop, mid_transit, dur = find_time_range(table_start, table_stop, epoch, period, dur, name=hdulist[1].header["OBJECT"])
        # When find_time_range fails, you know you are fucked.
        if start is None:
            return None, None
        # Strip out the rows that are actually transts.
        rows = strip_rows(hdulist[1], start, stop)
        # Also generate the negative samples
        neg_rows = strip_rows_negative(hdulist[1], start, stop)
        # Extract information of whether an eclipsing binary is being simulated
        is_eb = injected_table["EB_injection"][injected_i]
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
    injected = injection.parse_injected_table("/mnt/data/meta/kplr_dr25_inj1_plti.txt")
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
            if t is not None:
                ascii.write(t, root + "_quicklook.ecsv", format='ecsv')
            if neg_t is not None:
                ascii.write(neg_t, root + "_quicklook_negative.ecsv", format='ecsv')
        except Exception as e:
            logging.error("Error while processing file %s: %s", i, e)
            logging.error("Traceback: %s", traceback.format_exc())

if __name__ == "__main__":
    logging.basicConfig(filename="output.log", level=logging.INFO)
    main()
