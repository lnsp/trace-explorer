# Trace Explorer

This repository contains the source code of Trace Explorer, a toolset to explain and visualize database workload traces and benchmark datapoints.

## Usage

The first step in exploring your measurements is data preparation. Trace Explorer assist you in a multitude of ways, by automatically exploring different strategies to maximize dataset variance.

### Step 0: Converting the data into a common format

First, you have to convert your data into a common format. We use parquet for storing datasets because of its widespread compatibility and integrated compression. Each measurement must be converted into one row in the common dataset format. We provide the `CommonTraceConverter` interface to allow users to provide their own format converter. You can find examples for Snowset, Snowflake profiles, MSSQL and PostgreSQL in our `converter/` directory.

```bash
python3 -m trace-explorer convert --using myconverter.py --source mydataset/ --output mydatasetcommon.parquet
```

To speed up processing, we use [https://duckdb.com](DuckDB) and Parquet for storing intermediate data.

### Step 1: Find a good preprocessing pipeline

To maximize the possibility of being able to derive conclusions from the data, a good preprocessing pipeline is very necessary. We provide a set of common preprocessing primitives, and allow for automatic tuning by optimizing for global variance.