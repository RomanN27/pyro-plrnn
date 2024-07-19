import re
def get_time_stamp(site_name: str):

    time_stamp = site_name.split("_")[1]
    time_stamp = int(time_stamp.groups()[0]) if time_stamp else time_stamp
    return time_stamp