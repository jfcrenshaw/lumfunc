import jax
from dsps.dust.att_curves import (
    _frac_transmission_from_k_lambda,
    _get_eb_from_delta,
    noll09_k_lambda,
)


@jax.jit
def dust(wavelen, Av, delta):
    # Calculate UV bump amplitude from the power-law slope
    # uses formula from Kriek & Conroy 2013
    Eb = _get_eb_from_delta(delta)

    # Calculate k_lambda using the Noll 2009 model
    klambda = noll09_k_lambda(
        wave_micron=wavelen / 1e4,  # Angstroms -> microns
        uv_bump_ampl=Eb,
        plaw_slope=delta,
    )

    # Calculate fraction of transmitted flux
    return _frac_transmission_from_k_lambda(klambda, Av)
