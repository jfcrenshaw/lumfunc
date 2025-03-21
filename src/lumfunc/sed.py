from pathlib import Path

import astropy.units as u
import galsim
import numpy as np
from astropy.cosmology import Planck18
from eazy.sps import KC13
from lbg_tools import IGM

from .emission_lines import EmissionLineModel


class Bandpasses:
    def __init__(self, files: dict[str, str | Path]) -> None:
        # Read bandpass files
        self.bands = {
            band: galsim.Bandpass(file, "A").withZeropoint("AB")
            for band, file in files.items()
        }

    @property
    def wavelen(self) -> np.ndarray:
        return np.array(
            [band.effective_wavelength * 10 for band in self.bands.values()]
        )

    @property
    def zero_point(self) -> np.ndarray:
        return np.array([band.zeropoint for band in self.bands.values()])


class SED:
    def __init__(
        self,
        path: str,
        bandpasses: Bandpasses,
        redshift: float = 0.0,
        M1500: float = -20,
        dust_params: dict | None = None,
        igm_params: dict | None = None,
        emission_line_params: dict | None = None,
    ) -> None:
        # Load SED from file; (angstroms, flambda)
        sed = galsim.SED(path, "A", "flambda", fast=False)

        # Normalize SED at 1500 angstroms
        self._band_15000 = galsim.Bandpass(
            lambda wave: 1,
            "A",
            blue_limit=1500 - 50,
            red_limit=1500 + 50,
        ).withZeropoint("AB")
        self.M1500 = M1500
        self.sed_intrinsic = sed.withMagnitude(self.M1500, self._band_15000)

        # Set other parameters
        self.bandpasses = bandpasses
        self.redshift = redshift
        self.dust_params = {"Av": 0.0, "delta": 0.0} | (
            {} if dust_params is None else dust_params
        )
        self.igm_params = {"model": "inoue", "scale": 1.0, "min_z": 1.6} | (
            {} if igm_params is None else igm_params
        )
        self.emission_line_params = {"scale": 1.0, "fwhm": 20.0} | (
            {} if emission_line_params is None else emission_line_params
        )

    @property
    def dust_params(self) -> dict:
        return self._dust_params

    @dust_params.setter
    def dust_params(self, params: dict) -> None:
        params = {"Av": 0.0, "delta": 0.0} | params
        self.dust_model = KC13(**params)
        self._dust_params = params

    @property
    def igm_params(self) -> dict:
        return self._igm_params

    @igm_params.setter
    def igm_params(self, params: dict) -> None:
        params = {"model": "inoue", "scale": 1.0, "min_z": 2.0} | params
        sub_params = params.copy()
        sub_params.pop("min_z")
        self.igm_model = IGM(**sub_params)
        self._igm_params = params

    @property
    def emission_line_params(self) -> dict:
        return self._emission_line_params

    @emission_line_params.setter
    def emission_line_params(self, params: dict) -> None:
        params = {"scale": 1.0, "fwhm": 20.0} | params
        self.emission_line_model = EmissionLineModel(**params)
        self._emission_line_params = params

    @property
    def sed_observed(self) -> galsim.SED:
        # Get the intrinsic SED
        sed = self.sed_intrinsic

        # Apply dust
        if self.dust_params["Av"] > 0:
            dust = galsim.SED(self.dust_model.attenuate, "A", "1", fast=False)
            sed = sed * dust

        # Add emission lines
        if self.emission_line_params["scale"] > 0:
            sed = self.emission_line_model.add_lines(sed)

        # Redshift the SED
        z = self.redshift
        if z > 0:
            sed = sed.atRedshift(z)
            dL = np.max((Planck18.luminosity_distance(z).to(u.pc).value, 10))
            sed *= (10 / dL) ** 2 / (1 + z)

        # Apply IGM
        if z > self.igm_params["min_z"] and self.igm_params["scale"] > 0:
            igm = galsim.SED(
                lambda wavelen: self.igm_model.transmission(wavelen, z=self.redshift),
                "A",
                "1",
                fast=False,
            )
            sed = sed * igm

        return sed

    def eval(
        self,
        wavelen: np.ndarray | float,
        return_type: str = "mag",
    ) -> np.ndarray | float:
        # Make sure wavelen is an array
        wavelen = np.atleast_1d(wavelen)

        # Evaluate the flambda
        sed = self.sed_observed
        flambda = np.zeros_like(wavelen)
        idx = np.where((wavelen / 10 > sed.blue_limit) & (wavelen / 10 < sed.red_limit))
        flambda[idx] = sed(wavelen[idx] / 10)

        # Attach units
        flambda *= u.photon / u.nm / u.s / u.cm**2

        if return_type == "flambda":
            out = flambda.to(u.photon / u.AA / u.s / u.cm**2)
        elif return_type == "fnu":
            out = flambda.to(u.Jy, equivalencies=u.spectral_density(wavelen * u.AA))
        elif return_type == "mag":
            out = flambda.to(u.ABmag, equivalencies=u.spectral_density(wavelen * u.AA))

        return out.value

    @property
    def fluxes(self) -> np.ndarray:
        # Get observed SED
        sed = self.sed_observed

        # Calculate fluxes
        fluxes = np.array(
            [sed.calculateFlux(bp) for bp in self.bandpasses.bands.values()]
        )

        # Clip negative values
        fluxes = np.clip(fluxes, 0, None)

        return fluxes

    @property
    def mags(self) -> np.ndarray:
        # Manually calculate magnitudes from zero-clipped fluxes
        with np.errstate(divide="ignore"):
            return -2.5 * np.log10(self.fluxes) + self.bandpasses.zero_point
