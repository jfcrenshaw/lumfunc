"""Functions for calculation of IGM optical depth and transmission."""

import jax
import jax.numpy as jnp

from ._igm_inoue import igm_tau as inoue_tau
from ._igm_madau import igm_tau as madau_tau


@jax.jit
def igm_tau(
    wavelen: jax.Array,
    z: jax.Array,
    scale: float | jax.Array,
    model: int | jax.Array,
) -> jax.Array:
    """Class for calculation of IGM optical depth.

    Implements models of
    - Madau 1995: https://ui.adsabs.harvard.edu/abs/1995ApJ...441...18M/abstract
    - Inoue 2014: https://ui.adsabs.harvard.edu/abs/2014MNRAS.442.1805I/abstract

    Also allows for re-scaling of IGM optical depth, inspired by results of
    Thomas 2021: https://ui.adsabs.harvard.edu/abs/2021A%26A...650A..63T/abstract

    Parameters
    ----------
    wavelen : jax.Array
        Observed wavelength in Angstroms
    z : jax.Array
        Redshift
    scale : float or jax.Array
        Scaling applied to IGM optical depth calculated using the model.
    model : int or jax.Array
        Which IGM model to use. 0 corresponds to the Inoue model,
        1 corresponds to the Madau model.

    Returns
    -------
    jax.Array
        IGM optical depth
    """
    tau = jnp.where(
        model == 0,
        inoue_tau(wavelen, z),
        madau_tau(wavelen, z),
    )

    return tau * scale


@jax.jit
def igm_transmission(
    wavelen: jax.Array,
    z: jax.Array,
    scale: float | jax.Array,
    model: int | jax.Array,
) -> jax.Array:
    """Model for calculation of IGM transmission.

    Implements models of
    - Madau 1995: https://ui.adsabs.harvard.edu/abs/1995ApJ...441...18M/abstract
    - Inoue 2014: https://ui.adsabs.harvard.edu/abs/2014MNRAS.442.1805I/abstract

    Also allows for re-scaling of IGM optical depth, inspired by results of
    Thomas 2021: https://ui.adsabs.harvard.edu/abs/2021A%26A...650A..63T/abstract

    Parameters
    ----------
    wavelen : jax.Array
        Observed wavelength in Angstroms
    z : jax.Array
        Redshift
    scale : float or jax.Array
        Scaling applied to IGM optical depth calculated using the model.
    model : int or jax.Array
        Which IGM model to use. 0 corresponds to the Inoue model,
        1 corresponds to the Madau model.

    Returns
    -------
    jax.Array
        IGM optical depth
    """
    return jnp.exp(-igm_tau(wavelen, z, scale, model))
