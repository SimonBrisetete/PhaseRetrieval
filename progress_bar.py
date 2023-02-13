import sys
import time
import numpy as np


def progressbar(it, prefix="", size=60, out=sys.stdout):
    """
    Display a progress bar inside the console
    :param it: number of total iteration
    :type it:
    :param prefix: text displayed
    :type prefix:
    :param size: number of dots to be filled to complete the progress bar
    :type size:
    :param out: output address. Terminal by default
    :type out:
    :return:
    :rtype:
    """
    count = len(it)

    def show(j, elapsed_time):
        x = int(size*j/count)
        print("{}[{}{}] {}/{}; {} sec".format(prefix, "#"*x, "."*(size-x), j, count, np.round(elapsed_time,3)),
              end='\r', file=out, flush=True)
    show(0, 0)
    t0_ns = time.time_ns()
    for i, item in enumerate(it):
        yield item
        elapsed_time = (time.time_ns() - t0_ns)*1e-9
        show(i+1, elapsed_time)
    print("\n", flush=True, file=out)
