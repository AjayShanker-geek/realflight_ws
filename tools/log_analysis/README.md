# Log Analysis Tools

Quick analysis helpers for L2C logs.

## Summary (controller log)

```
python3 tools/log_analysis/l2c_log_summary.py /path/to/l2c_log_*.csv
```

Optional JSON output:

```
python3 tools/log_analysis/l2c_log_summary.py /path/to/l2c_log_*.csv --json /tmp/l2c_summary.json
```

## Summary (debug log)

```
python3 tools/log_analysis/l2c_debug_summary.py /path/to/l2c_debug_*.csv
```

## Plots

```
python3 tools/log_analysis/l2c_plot_log.py /path/to/l2c_log_*.csv
```
