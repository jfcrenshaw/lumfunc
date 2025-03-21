import astropy.units as u
import galsim
import numpy as np

from .utils import data_dir


class EmissionLineModel:

    lines = dict(
        Lya=1216 * u.AA,
        OII_doublet=3727 * u.AA,
        Hgamma=4340 * u.AA,
        Hbeta=4861 * u.AA,
        OIII_4959=4959 * u.AA,
        OIII_5007=5007 * u.AA,
        NII_6548=6548 * u.AA,
        Halpha=6563 * u.AA,
        NII_6584=6584 * u.AA,
        SII_6716=6716 * u.AA,
        SII_6731=6731 * u.AA,
    )

    def __init__(self, scale: float = 1.0, fwhm: float = 20.0) -> None:
        # Load parameters for JAX-Net
        self.jn_model_dict = np.load(
            data_dir / "emission_line_model_params.npy",
            allow_pickle=True,
        ).item()

        # Create narrow continuum bands for emission lines
        self.continuum_bands = dict()
        for line, wavelen in self.lines.items():
            wavelen = wavelen.value
            bp = galsim.Bandpass(
                lambda wave: 1,
                "A",
                blue_limit=wavelen - 50,
                red_limit=wavelen + 50,
            )
            bp = bp.withZeropoint("AB")
            self.continuum_bands[line] = bp

        # Create medium-bands for JAX-Net
        bins = np.linspace(3400, 7000, 13)
        mid = np.mean((bins[:-1], bins[1:]), axis=0).astype(int)
        jn_bands = dict()
        for i in range(len(bins) - 1):
            bp = galsim.Bandpass(
                lambda wave: 1,
                "A",
                blue_limit=bins[i],
                red_limit=bins[i + 1],
            )
            bp = bp.withZeropoint("AB")
            jn_bands[int(mid[i])] = bp
        self.jn_bins = bins
        self.jn_bin_width = np.diff(np.linspace(3400, 7000, 13))[0]
        self.jn_bands = jn_bands

        # Save other parameters
        self.scale = scale
        self.fwhm = fwhm

    @staticmethod
    def _feedforward_predict(params, x):
        # Parameters for jax selu activation function
        # (this is so I can replace all jax with numpy)
        alpha = 1.6732632423543772848170429916717
        scale = 1.0507009873554804934193349852946

        activations = x
        for w, b in params[:-1]:
            # Pass through layer
            outputs = np.dot(w, activations) + b

            # Apply selu
            activations = scale * np.where(
                outputs > 0,
                outputs,
                alpha * np.exp(outputs) - alpha,
            )

        # Pass through the final layer without selu
        w_final, b_final = params[-1]
        final_outputs = np.dot(w_final, activations) + b_final

        return final_outputs[0]

    def continuum_fluxes(self, sed: galsim.sed) -> dict:
        return {
            line: sed.calculateFlux(bp) for line, bp in self.continuum_bands.items()
        }

    def jn_colors(self, sed: galsim.sed) -> np.ndarray:
        # Calculate fluxes
        flambda_avgs = []
        for wave, bp in self.jn_bands.items():
            flux = sed.calculateFlux(bp) * u.photon / u.s / u.cm**2
            flambda_avg = flux / (self.jn_bin_width * u.AA)
            flambda_avg = (
                flambda_avg.to(
                    u.erg / u.AA / u.s / u.cm**2,
                    equivalencies=u.spectral_density(wave * u.AA),
                )
                * 1e17
            )  # Convert to units in Fig. 2 from Khederlarian 2024

            # I am calculating everything in the z=0 frame
            # Ashod's formula assumes these fluxes haven't been corrected
            # for redshift/luminosity distance scaling
            # So I can ignore these factors, EXCEPT he doesn't divide his
            # luminosity distance by 10 pc, and uses Mpc, so there is a
            # residual factor of (10 pc / Mpc)^2 = 1e-10
            flambda_avg *= 1e-10

            flambda_avgs.append(flambda_avg.value)
        flambda_avgs = np.array(flambda_avgs)

        # Construct colors
        colors = np.log10(flambda_avgs[1:] / flambda_avgs[:-1])  # type: ignore

        return colors

    def _jn_predict(self, sed: galsim.sed) -> dict:

        # Get the colors
        colors = self.jn_colors(sed)

        # Predict equivalent widths
        model_dict = self.jn_model_dict
        results = {}
        for line in model_dict:
            features = (colors - model_dict[line]["mean"]) / model_dict[line]["scale"]
            ew = np.sinh(
                self._feedforward_predict(model_dict[line]["params"], features)
            )
            results[line] = float(ew)

        return results

    def predict(self, sed: galsim.sed) -> dict:
        # Predict equivalent widths using the network
        ews = self._jn_predict(sed)

        # Calculate continuum flux at each line
        continuum = self.continuum_fluxes(sed)

        # Calculate Lyman-alpha equivalent width
        # using model fit to Lya/Ha vs Ha/Hb values from Fig. 12 of
        # https://arxiv.org/abs/1505.07483
        CaWa = continuum["Halpha"] * ews["Halpha"]
        CbWb = continuum["Hbeta"] * ews["Hbeta"]
        Cl = continuum["Lya"]
        Wl = 170.2 * CaWa / Cl * (CaWa / CbWb) ** -3.46
        ews["Lya"] = Wl

        # Apply the scales
        ews = {line: self.scale * ew for line, ew in ews.items()}

        return ews

    def add_lines(self, sed: galsim.sed) -> galsim.sed:
        # Predict equivalent widths
        ews = self.predict(sed)

        # Calculate continuum flux
        cont_flux = self.continuum_fluxes(sed)

        # Convert to flux density (divide by 100 Angstroms)
        cont_flux_density = {line: flux / 100 for line, flux in cont_flux.items()}

        # Calculate line fluxes
        line_flux = {line: cont_flux_density[line] * ews[line] for line in ews}

        # Add lines to the sed
        for line in line_flux:
            sed = sed + galsim.EmissionLine(
                self.lines[line].value,
                flux=line_flux[line],
                fwhm=self.fwhm,
                wave_type="A",
            )

        return sed
