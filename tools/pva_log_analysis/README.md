# PVA Log Analysis Tools

Helpers for `pva_control` logs.

## Summary (controller log)

```
python3 tools/pva_log_analysis/pva_log_summary.py /path/to/pva_log_*.csv
```

Optional JSON output:

```
python3 tools/pva_log_analysis/pva_log_summary.py /path/to/pva_log_*.csv --json /tmp/pva_summary.json
```

## Summary (debug log)

```
python3 tools/pva_log_analysis/pva_debug_summary.py /path/to/pva_debug_*.csv
```

## Plots

Single log:

```
python3 tools/pva_log_analysis/pva_plot_log.py /path/to/pva_log_*.csv
```

SITL multi-drone (pass base dir or run dir):

```
python3 tools/pva_log_analysis/pva_plot_log.py /path/to/sitl_log
```

Plots include position, acceleration norm, feedback norms (if present), and jerk (commanded vs actual when available).
