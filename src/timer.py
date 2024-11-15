import time, datetime

class Timer:
  def set(self):
    self.start = time.perf_counter()
  def get(self):
    return datetime.timedelta(seconds=(time.perf_counter() - self.start))
