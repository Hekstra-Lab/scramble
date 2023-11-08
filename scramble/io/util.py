
def get_first_key_of_type(ds, dtype):
    idx = ds.dtypes == dtype
    keys = ds.dtypes.keys()[idx]
    key = keys[0]
    return key


