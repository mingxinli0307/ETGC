#!/usr/bin/env python3
"""Run the DBLP large-OrthQA follow-up using the four-loss tuning runner."""

import run_school_four_loss_tuning as tuning


# Continue the completed OrthQA axis beyond 20 without rerunning the other
# one-factor axes.  All remaining settings are inherited from the shared
# runner and are validated before each subprocess starts.
tuning.ORTH_VALUES = (30.0, 50.0, 100.0, 200.0)
tuning.NCUT_VALUES = ()
tuning.PROX_VALUES = ()
tuning.ESG_VALUES = ()


if __name__ == "__main__":
    raise SystemExit(tuning.main())
