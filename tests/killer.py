import multiprocessing as mp
import time
import os


def __interrupt(pid, delay):
    time.sleep(delay)
    os.kill(pid, 2)  # interrupt


def interrupt(delay):
    pid = os.getpid()
    proc = mp.Process(target=__interrupt, args=(pid, delay))
    proc.start()
