# type: ignore
import builtins
import inspect
import pprint
import re


def pp(*args, **kwargs):
    """PrettyPrint function that prints the variable name when available and pprints the data."""
    name = None

    # Fetch the current frame from the stack.
    frame = inspect.currentframe().f_back

    # Prepare the frame info
    frame_info = inspect.getframeinfo(frame)

    # walk thru the lines of the function
    for line in frame_info[3]:
        # search for the p() function call with a fancy regexp  TODO: wtf gtf away from regexp
        m = re.search(r"\bpp\s*\([^)]*)\s*\)", line)
        if m:
            # print("# %s:" % m.group(1), end = ' ')
            print(f"# {m.group(1)}:", end = ' ')
        break

    pprint.pprint(*args, **kwargs)

builtins.pf = pprint.pformat
builtins.pp = pp

"""
This is all much too hacky for prod, but its still useful when working on a large project where you need print statements to debug.
Alternative (and better) will be added later.
"""