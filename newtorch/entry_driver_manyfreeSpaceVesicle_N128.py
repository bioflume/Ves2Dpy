"""Deprecated: use entry_vesnet.py with resolution 128 in vesnet_config.json."""

import warnings

warnings.warn(
    "entry_driver_manyfreeSpaceVesicle_N128.py is deprecated; run entry_vesnet.py instead.",
    DeprecationWarning,
    stacklevel=1,
)

from entry_vesnet import main

if __name__ == "__main__":
    main()
