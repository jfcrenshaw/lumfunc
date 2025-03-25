from itertools import compress, product
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from .utils import data_dir


class Config:
    def __init__(self, path: str | Path) -> None:
        with open(path, "r") as file:
            config = yaml.safe_load(file)
        self.raw = config

        _data_dir = config.get("data_dir", data_dir)

        self.filter_files = {
            band: Path(_data_dir) / Path(file)
            for band, file in config["filter_files"].items()
        }

        self.sed_files = {
            i: Path(_data_dir) / Path(file) for i, file in config["sed_files"].items()
        }
        self.sed_list = list(self.sed_files.values())

        self.z_grid = self._unpack_range("z_grid")
        self.M_grid = self._unpack_range("M_grid")

        self.dust_seds = self._unpack_range("dust", "seds", allows_step=False)
        self.dust_delta = self._ensure_list(config["dust"]["delta"])
        self.dust_Av = self._ensure_list(config["dust"]["Av"])

        self.line_seds = self._unpack_range("emission_lines", "seds", allows_step=False)
        self.line_scale = self._ensure_list(config["emission_lines"]["scale"])
        self.line_fwhm = self._ensure_list(config["emission_lines"]["fwhm"])

        self.igm_model = self._ensure_list(config["igm"]["model"])
        self.igm_scale = self._ensure_list(config["igm"]["scale"])
        self.igm_min_z = config["igm"]["min_z"]

    @staticmethod
    def _ensure_list(value) -> list:
        return np.atleast_1d(value).tolist()

    def _unpack_range(self, *key, allows_step: bool = True) -> list:
        # Get down to the grid config
        param = self.raw
        for k in key:
            param = param[k]

        # Try every type option
        if isinstance(param, float) or isinstance(param, int):
            return self._ensure_list(param)
        elif isinstance(param, list):
            return param
        elif isinstance(param, dict):
            step = 1 if not allows_step else param["step"]
            return np.arange(
                param["min"],
                param["max"] + step,
                step,
            ).tolist()
        else:
            msg = f"{key} must be a number, list, or dictionary with keys min,max"
            msg = msg + ",step" if allows_step else msg
            raise TypeError(msg)

    @staticmethod
    def _prune_zero_models(grid, columns):
        idx = []
        for _, group in grid.groupby("template"):
            # Find where the first column is zero
            mask0 = np.isclose(group[columns[0]], 0)

            # Get value of second column closest to zero
            val = np.abs(group.loc[mask0, columns[1]]).min()

            # Find where second column deviates from this value
            mask1 = ~np.isclose(group[columns[1]], val)

            # Remove rows where the first column is close to zero, but the second
            # column deviates from it's min absolute value
            idx.extend(group[mask0 & mask1].index.to_list())

        return grid.drop(idx)

    @property
    def grid(self) -> pd.DataFrame:
        grid = []
        for template in self.sed_files:
            # IGM params
            igm_model = [0 if model == "inoue" else 1 for model in self.igm_model]
            igm_scale = self.igm_scale

            # Decide on dust params
            if template in self.dust_seds:
                dust_Av = self.dust_Av
                dust_delta = self.dust_delta
            else:
                dust_delta = [0.0]
                dust_Av = [0.0]

            # Decide on emission lines
            if template in self.line_seds:
                line_scale = self.line_scale
                line_fwhm = self.line_fwhm
            else:
                line_scale = [0.0]
                line_fwhm = [0.0]

            # Add models to the grid
            grid += list(
                product(
                    [template],
                    igm_model,
                    igm_scale,
                    dust_delta,
                    dust_Av,
                    line_scale,
                    line_fwhm,
                    self.z_grid,
                )
            )

        grid = pd.DataFrame(
            grid,
            columns=[
                "template",
                "igm_model",
                "igm_scale",
                "dust_delta",
                "dust_Av",
                "line_scale",
                "line_fwhm",
                "redshift",
            ],
        )

        # When any of line_scale, dust_Av, igm_scale are zero
        # we don't need to keep the full outer product of these values
        # with the other dust/line/igm parameters
        grid = self._prune_zero_models(grid, ["line_scale", "line_fwhm"])
        grid = self._prune_zero_models(grid, ["dust_Av", "dust_delta"])
        grid = self._prune_zero_models(grid, ["igm_scale", "igm_model"])

        return grid
