"""Implementation of the IGM model from Madau 1995

https://ui.adsabs.harvard.edu/abs/1995ApJ...441...18M/abstract
"""

from functools import partial

import jax
import jax.numpy as jnp

# Wavelengths and coefficients for equation 12, 15
# values of higher order terms retrieved from FSPS:
# https://github.com/cconroy20/fsps/blob/master/src/igm_absorb.f90#L63-L64
_tls_wavelen = jnp.array(
    [
        1215.67,
        1025.72,
        972.537,
        949.743,
        937.803,
        930.748,
        926.226,
        923.150,
        920.963,
        919.352,
        918.129,
        917.181,
        916.429,
        915.824,
        915.329,
        914.919,
        914.576,
    ]
)
_tls_coeff = jnp.array(
    [
        0.0036,
        0.0017,
        0.0011846,
        0.0009410,
        0.0007960,
        0.0006967,
        0.0006236,
        0.0005665,
        0.0005200,
        0.0004817,
        0.0004487,
        0.0004200,
        0.0003947,
        0.000372,
        0.000352,
        0.0003334,
        0.00031644,
    ]
)


@partial(jax.vmap, in_axes=(1, None), out_axes=1)
def tls(wavelen: jax.Array, z: jax.Array) -> jax.Array:
    """Calculate optical depth contribution from Lyman-series

    Parameters
    ----------
    wavelen : jax.Array
        Observed wavelength in Angstroms
    z : jax.Array
        Redshift

    Returns
    -------
    jax.Array
        Optical depth contribution
    """
    # Evaluate power-law terms at every wavelength
    w = wavelen[:, None] / _tls_wavelen[None, :]
    tau = _tls_coeff * w**3.46

    # Mask values outside appropriate wavelength ranges
    mask = (w >= 1) & (w <= 1 + z)
    tau = jnp.where(mask, tau, 0)

    # Sum over every line in the series
    tau = tau.sum(axis=-1)

    return tau


def tlc(wavelen: jax.Array, z: jax.Array) -> jax.Array:
    """Calculate optical depth contribution from Lyman-continuum

    Using approximation of Eq. 16 from Madau 1995. See footnote 3.

    This private function calculates the Inoue polynomials, which are then
    patched at low wavelengths in the public function.

    Parameters
    ----------
    wavelen : jax.Array
        Observed wavelength in Angstroms
    z : jax.Array
        Redshift

    Returns
    -------
    jax.Array
        Optical depth contribution
    """
    xc = wavelen / 911.75
    xm = 1 + z

    tau = (
        0.25 * xc**3 * (xm**0.46 - xc**0.46)
        + 9.4 * xc**1.5 * (xm**0.18 - xc**0.18)
        - 0.7 * xc**3 * (xc ** (-1.32) - xm ** (-1.32))
        - 0.023 * (xm**1.68 - xc**1.68)
    )

    mask = xc <= 1 + z
    tau = jnp.where(mask, tau, 0)

    return tau


@jax.jit
def igm_tau(wavelen: jax.Array, z: jax.Array) -> jax.Array:
    """Calculate optical depth of the IGM using the Madau model.

    This function contains a simple patch so that the continuum optical
    depth doesn't decrease at lower wavelengths.

    Parameters
    ----------
    wavelen : jax.Array
        Observed wavelength in Angstroms
    z : jax.Array
        Redshift

    Returns
    -------
    jax.Array
       IGM optical depth
    """
    # Calculate components and sum
    series = tls(wavelen, z)
    continuum = tlc(wavelen, z)
    total = series + continuum

    # Hack to fix low-wavelength continuum
    # i.e., continuum fitting function decreases at low-wavelengths, which isn't
    # expected. As continuum tau starts to decrease towards lower wavelength,
    # we simply set values to the max value

    # Mask for all points before point where series and continuum curves cross
    mask = jnp.cumsum((continuum > series)[:, ::-1], axis=1)[:, ::-1].astype(bool)

    # Replace with cumulative max (moving towards smaller wavelengths),
    # but only for points where continuum is greater than the series
    tau = jnp.where(mask, jax.lax.cummax(total, reverse=True, axis=1), total)

    return tau
