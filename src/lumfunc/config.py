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

        self.z_grid = self._unpack_grid("z_grid")
        self.M_grid = self._unpack_grid("M_grid")

        self.dust_seds = self._unpack_grid("dust", "seds", allows_step=False)
        self.dust_delta = self._ensure_list(config["dust"]["delta"])
        self.dust_Av = self._ensure_list(config["dust"]["Av"])

        self.line_seds = self._unpack_grid("emission_lines", "seds", allows_step=False)
        self.line_scale = self._ensure_list(config["emission_lines"]["scale"])
        self.line_fwhm = self._ensure_list(config["emission_lines"]["fwhm"])

        self.igm_model = self._ensure_list(config["igm"]["model"])
        self.igm_scale = self._ensure_list(config["igm"]["scale"])
        self.igm_min_z = config["igm"]["min_z"]

    @staticmethod
    def _ensure_list(value) -> list:
        return np.atleast_1d(value).tolist()

    def _unpack_grid(self, *key, allows_step: bool = True) -> list:
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

    @property
    def grid(self) -> pd.DataFrame:
        grid = []
        for template in self.sed_files:
            # Split redshift grid into pre-/post-IGM
            mask = np.less(self.z_grid, self.igm_min_z)
            z_pre_igm = compress(self.z_grid, mask)
            z_post_igm = compress(self.z_grid, ~mask)
            igm_model = self.igm_model
            igm_scale = self.igm_scale

            # Decide on dust params
            if template in self.dust_seds:
                dust_delta = self.dust_delta
                dust_Av = self.dust_Av
            else:
                dust_delta = [0]
                dust_Av = [0]

            # Decide on emission lines
            if template in self.line_seds:
                line_scale = self.line_scale
                line_fwhm = self.line_fwhm
            else:
                line_scale = [0]
                line_fwhm = [0]

            # First the pre-igm grid
            grid += list(
                product(
                    [template],
                    [igm_model[0]],
                    [1],
                    dust_delta,
                    dust_Av,
                    line_scale,
                    line_fwhm,
                    z_pre_igm,
                )
            )
            # Now the post-igm grid
            grid += list(
                product(
                    [template],
                    igm_model,
                    igm_scale,
                    dust_delta,
                    dust_Av,
                    line_scale,
                    line_fwhm,
                    z_post_igm,
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

        return grid
