import astropy.units as u
import galsim
import jax
import jax.numpy as jnp
import jax_cosmo as jc
import numpy as np

from .dust import dust
from .emission_lines import EmissionLineModel
from .igm import igm_transmission


def load_trim_thin(file, w_min, w_max):
    # Load
    x, y = np.genfromtxt(file, unpack=True)

    # Trim
    idx = np.searchsorted(x, [w_min, w_max])
    x = x[idx[0] : idx[1]]
    y = y[idx[0] : idx[1]]

    # Thin
    x, y = galsim.utilities.thin_tabulated_values(x, y)

    return x, y


def load_files_from_config(config, w_min, w_max):
    sed_dict = {}
    for N, file in config.sed_files.items():
        sx, sy = load_trim_thin(file, w_min, w_max)
        sed_dict[N] = {"x": sx, "y": sy}

    band_dict = {}
    for band, file in config.filter_files.items():
        bx, by = load_trim_thin(file, w_min, w_max)
        by = bx * by / jnp.trapezoid(bx * by, x=bx)
        zero_point = 2.5 * np.log10(jnp.trapezoid(bx**-2 * by, x=bx))
        band_dict[band] = {"x": bx, "y": by, "zero_point": zero_point}

    return sed_dict, band_dict


def add_emission_lines(sx, sy, line_scale, line_fwhm):
    """For now, piggy-back off my previous galsim-based infrastructure"""
    # Create a galsim SED object for the original SED
    sed = galsim.SED(
        galsim.LookupTable(sx, sy, interpolant="linear"),
        wave_type="A",
        flux_type="flambda",
    )

    # Add emission lines using the model
    elm = EmissionLineModel(scale=line_scale, fwhm=line_fwhm)
    sed = elm.add_lines(sed)

    # Extract model on grid
    sx = 10 * sed.wave_list  # Angstroms
    sy = sed(sx / 10) * u.photon / u.nm / u.s / u.cm**2
    sy = sy.to(
        u.erg / u.AA / u.s / u.cm**2, equivalencies=u.spectral_density(sx * u.AA)
    )

    return sx, sy


def luminosity_distance(z: jax.Array, cosmo: jc.Cosmology) -> jax.Array:
    fk = jc.background.transverse_comoving_distance(cosmo, 1 / (1 + z))  # Mpc/h
    fk = fk * 1e6 / cosmo.h  # pc

    dL = (1 + z) * fk

    dL = jnp.clip(dL, 10, None)

    return dL


@jax.jit
def _create_transmission(
    sx,
    redshift,
    dust_Av,
    dust_delta,
    igm_model,
    igm_scale,
    cosmo,
):
    # Create the transmission matrix
    # Each row represents a redshift
    # Each column represents a rest-frame wavelength
    transmission = jnp.ones((redshift.size, sx.size))

    # Multiply dust
    transmission *= dust(sx[None, :], dust_Av, dust_delta)

    # Multiply IGM transmission
    transmission *= igm_transmission(
        sx[None, :] * (1 + redshift), redshift, igm_scale, igm_model
    )

    # Multiply by the redshift factor
    dL = luminosity_distance(redshift, cosmo)
    transmission *= (10 / dL) ** 2 / (1 + redshift)

    return transmission


def create_transmission(sx, grid, cosmo):
    redshift, dust_Av, dust_delta, igm_model, igm_scale = jnp.hsplit(
        grid[
            [
                "redshift",
                "dust_Av",
                "dust_delta",
                "igm_model",
                "igm_scale",
            ]
        ].values,
        5,
    )

    transmission = _create_transmission(
        sx=sx,
        redshift=redshift,
        dust_Av=dust_Av,
        dust_delta=dust_delta,
        igm_model=igm_model,
        igm_scale=igm_scale,
        cosmo=cosmo,
    )

    return redshift, transmission


@jax.jit
def _calc_mags(
    sx,
    sy,
    bx,
    by,
    bzp,
    redshift,
    transmission,
):
    # Multiply bandpass transmission
    trans = transmission * jnp.interp(sx * (1 + redshift), bx, by, left=0, right=0)

    # Calculate fluxes
    fluxes = jnp.trapezoid(trans * sy, sx, axis=1)

    # Calculate magnitudes
    mags = -2.5 * jnp.log10(fluxes) + bzp

    return mags


def calc_mags(sx, sy, band_dict, redshift, transmission):
    # (note it turns out this loop is faster than the vectorized version)
    # (actually maybe I was just fooled by timing due to asynchronous dispatch?
    #  I might want to re-investigate this as some point)
    mags = []
    for band in band_dict.values():
        mags.append(
            _calc_mags(
                sx=sx,
                sy=sy,
                bx=band["x"],
                by=band["y"],
                bzp=band["zero_point"],
                redshift=redshift,
                transmission=transmission,
            )
        )
    mags = jnp.array(mags).T

    return mags
