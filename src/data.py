"""Contains tools for initial data preprocessing and then data loading during training."""

from typing import Any

from pybaseball import statcast
import polars as pl
import torch
from torch.utils.data import Dataset
from logzero import logger
import morphers

MORPHER_MAPPING = {
    "categorical": morphers.Integerizer,
    "numeric": morphers.Normalizer,
}

label_mapping = {
    "catcher_interf": 0,
    "double": 0,
    "sac_bunt": 1,
    "triple_play": 1,
    "fielders_choice_out": 1,
    "fielders_choice": 1,
    "hit_by_pitch": 0,
    "truncated_pa": 0,
    "sac_fly": 1,
    "field_out": 1,
    "strikeout": 1,
    "single": 0,
    "double_play": 1,
    "triple": 0,
    "force_out": 1,
    "field_error": 0,
    "grounded_into_double_play": 1,
    "home_run": 0,
    "walk": 0,
    "sac_fly_double_play": 1,
    "None": 0,
    "strikeout_double_play": 1,
}

def pad_tensor_dict(tensor_dict: dict[str, torch.Tensor], max_length: int):
    """
    Pad a tensor dict up to the max length.
    Padded Location = 0
    Returns an Additive Mask
    """

    init_length = next(iter(tensor_dict.values())).shape[0]
    if init_length >= max_length:
        padded_tensor_dict = {k: v[:max_length] for k, v in tensor_dict.items()}
    else:
        padded_tensor_dict = {
            k: torch.nn.functional.pad(v, [0, max_length - init_length], value=0)
            for k, v in tensor_dict.items()
        }
    # FALSE IS NOT PAD, TRUE IS PAD

    pad_mask = torch.ones([max_length], dtype=torch.bool)
    pad_mask[: min(init_length, max_length)] = False
    pad_mask = torch.where(pad_mask, float("-inf"), 0.0)
    return padded_tensor_dict, pad_mask


def initial_prep(
    path: str,
    start_date: str,
    end_date: str,
) -> str:
    """Do one-time data preparation.
    Here we'll just download statcast data and save it.
    """

    init_data = statcast(start_dt=start_date, end_dt=end_date)
    init_data = pl.DataFrame(init_data)
    logger.info(f"Writing {len(init_data)} rows to {path}")
    init_data.write_parquet(path)
    return path


class TrainingDataset(Dataset):
    """Torch Dataset class for training."""

    def __init__(
        self,
        path: str,
        sequence_length: int,
        morpher_spec: (
            dict[str, tuple[type[morphers.base.base.Morpher], dict[str, Any]]] | None
        ) = None,
        prefit_morphers: dict[str, morphers.base.base.Morpher] | None = None,
    ):
        """Load a dataset and do any prep required.

        If at all possible, this should be deterministic, given a specific file.
        """
        super().__init__()
        self.sequence_length = sequence_length

        df = pl.read_parquet(path).with_columns(
            pl.col("pitcher")
            .cum_count()
            .over(
                partition_by="pitcher",
                order_by=["game_pk", "at_bat_number", "pitch_number"],
            )
            .alias("pitcher_pitch_number"),
            pl.col("batter")
            .cum_count()
            .over(
                partition_by="batter",
                order_by=["game_pk", "at_bat_number", "pitch_number"],
            )
            .alias("batter_pitch_number"),
        )

        # Initialize the morphers

        if morpher_spec is not None:

            # Map feature types to morphers.
            morpher_spec = {
                k: (MORPHER_MAPPING[v[0]], v[1]) for k, v in morpher_spec.items()
            }
            self.morphers = {
                column: morpher.from_data(df[column], **morpher_kwargs)
                for column, (morpher, morpher_kwargs) in morpher_spec.items()
            }

        else:
            self.morphers = prefit_morphers

        self.pitcher_morpher = morphers.Integerizer.from_data(df["pitcher"])

        # Create the target dataframe
        self.outcomes = df.select(
            self.pitcher_morpher(pl.col("pitcher")).alias("pitcher"),
            "batter",
            pl.col("events").replace(label_mapping, return_dtype=pl.Int64).alias("events"),
            pl.col("pitcher_pitch_number")
            .min()
            .over(["game_pk", "at_bat_number"])
            .alias("pitcher_pitch_number"),
            pl.col("batter_pitch_number")
            .min()
            .over(["game_pk", "at_bat_number"])
            .alias("batter_pitch_number"),
        ).drop_nulls()

        # Create the batter pitch sequence dataframe.
        self.pitches = df.select(
            "batter",
            "batter_pitch_number",
            *[morpher(pl.col(column)) for column, morpher in self.morphers.items()],
        ).sort("batter", "batter_pitch_number")

    def __len__(self) -> int:
        return self.outcomes.height

    def __getitem__(self, idx) -> dict[str, torch.Tensor]:
        """Return a single instance of the training dataset, as a dictionary."""

        target_row = self.outcomes.row(idx, named=True)

        target_info = {
            "pitcher": torch.tensor(target_row["pitcher"], dtype=torch.long),
            "target": torch.tensor(target_row["events"], dtype=torch.float),
        }

        pitches = self.pitches.filter(
            pl.col("batter") == target_row["batter"],
            pl.col("batter_pitch_number").is_between(
                target_row["batter_pitch_number"] - self.sequence_length,
                target_row["batter_pitch_number"],
                closed="left",
            ),
        )

        inputs, pad_mask = pad_tensor_dict(
            {
                k: torch.tensor(pitches[k], dtype=morpher.required_dtype)
                for k, morpher in self.morphers.items()
            },
            max_length=self.sequence_length,
        )

        return inputs | target_info, pad_mask

if __name__ == "__main__":

    ds = TrainingDataset(
        "../data/raw_data.parquet",
        256,
        {"release_speed": ("numeric", {})},
    )

    print(ds[590])
    