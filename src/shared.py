import time, datetime

def runtime(start):
    return datetime.timedelta(seconds=(time.time() - start))
