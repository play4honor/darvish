import polars as pl

df = pl.read_parquet("./data/raw_data.parquet").with_columns(
    pl.col("pitcher")
    .cum_count()
    .over(partition_by="pitcher", order_by=["game_pk", "at_bat_number", "pitch_number"])
    .alias("pitcher_pitch_number"),
    pl.col("pitcher")
    .cum_count()
    .over(partition_by="batter", order_by=["game_pk", "at_bat_number", "pitch_number"])
    .alias("batter_pitch_number"),
)

outcomes = df.select(
    "pitcher",
    "batter",
    "events",
    pl.col("pitcher_pitch_number")
    .min()
    .over(["game_pk", "at_bat_number"])
    .alias("pitcher_pitch_number"),
    pl.col("batter_pitch_number")
    .min()
    .over(["game_pk", "at_bat_number"])
    .alias("batter_pitch_number"),
).drop_nulls()


print(outcomes)

# Get
