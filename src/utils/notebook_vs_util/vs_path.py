import glob

from IPython import get_ipython
from loguru import logger
import os


def vs_notebook_name():
    """Return the name of the notebook.

    Returns
    -------
    str
        The name of the notebook.
    """
    ip = get_ipython()
    path = None
    if "__vsc_ipynb_file__" in ip.user_ns:
        logger.debug("Extract the notebook name")
        path = ip.user_ns["__vsc_ipynb_file__"]
        note_name = path.split("\\")[-1].split("ipynb")[0][:-1]
        logger.debug(f"Extract the notebook name: {note_name}")
        return note_name
    else:
        logger.error("Not working")
        return "erro"


def number_log(path_logs:str)->int:
    """
    Determine the next log number based on existing log files in the specified directory.

    Parameters
    ----------
    path_logs : str
        The directory path where log files are stored.

    Returns
    -------
    int
        The next log number to be used.

    Notes
    -----
    This function searches for log files in the specified directory that match the notebook name
    and have a ".log" extension. It then determines the highest log number and returns the next
    number in sequence. If no log files are found, it returns 1.

    Examples
    --------
    >>> number_log("/path/to/logs")
    5
    """
    note_name = vs_notebook_name()

    path_with_logs = os.path.join(path_logs, "*")
    logs = [x for x in glob.glob(f"{path_with_logs}") if note_name in x and "log" in x]
    if len(logs):
        n_ = max([int(x.split(f"{note_name}_")[-1].split(".log")[0]) for x in logs])
        n = n_ + 1

    else:
        n = 1

    logger.debug(f"The next log number determined is: {n}")

    return n
