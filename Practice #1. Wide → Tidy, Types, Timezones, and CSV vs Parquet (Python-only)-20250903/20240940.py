import csv, os
from zoneinfo import ZoneInfo

HOUR_COLS = [f"h{h:02d}" for h in range(8, 19)]  # h08..h18
LOCAL_TZ = ZoneInfo("Europe/London")
UTC_TZ = ZoneInfo("UTC")

DATA_IN = "data/sample_bike_wide.csv"
OUT_DIR = "out"
CSV_OUT = os.path.join(OUT_DIR, "bike_tidy.csv")
PARQUET_OUT = os.path.join(OUT_DIR, "bike_tidy.parquet")

### WRITE_YOUR_CODE
from datetime import datetime
import pyarrow as pa
import pyarrow.parquet as pq

# wide to tidy
MISSING = {"", "NA", "N/A"}
def parse_cnt(val):
    if val in MISSING:
        return None
    return int(val)

def wide_to_tidy():
    tidy_rows_csv = []
    tidy_rows_parquet = []

    with open(DATA_IN, newline = "", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            station_id = row["station_id"]
            station_name = row["station_name"]
            date = row["date"]

            for col in HOUR_COLS:
                cnt = parse_cnt(row[col])
                hour = int(col[1:])
    
                dt_local = datetime.fromisoformat(date).replace(
                    hour = hour, tzinfo=LOCAL_TZ
                )
                dt_utc = dt_local.astimezone(UTC_TZ)
                tidy_rows_csv.append([
                    station_id,
                    station_name,
                    dt_local.isoformat(),
                    dt_utc.isoformat().replace("+00:00", "Z"),
                    "None" if cnt == None else cnt
                ])

                tidy_rows_parquet.append([
                    station_id,
                    station_name,
                    dt_local.isoformat(),
                    dt_utc.isoformat().replace("+00:00", "Z"),
                    cnt
                ])


    # save csv
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(CSV_OUT, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["station_id", "station_name", "timestamp_local", "timestamp_utc", "count"])
        writer.writerows(tidy_rows_csv)

    # save parquet
    table = pa.Table.from_arrays(
        [
            [r[0] for r in tidy_rows_parquet],  # station_id
            [r[1] for r in tidy_rows_parquet],  # station_name
            [r[2] for r in tidy_rows_parquet],  # timestamp_local
            [r[3] for r in tidy_rows_parquet],  # timestamp_utc
            [r[4] for r in tidy_rows_parquet],  # count
        ],
        names = ["station_id", "station_name", "timestamp_local", "timestamp_utc", "count"]
    )
    pq.write_table(table, PARQUET_OUT)

if __name__ == "__main__":
    wide_to_tidy()