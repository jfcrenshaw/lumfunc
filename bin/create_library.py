import argparse

import jax_cosmo as jc
import pandas as pd
from schwimmbad import MultiPool

from lumfunc import (
    Config,
    add_emission_lines,
    calc_mags,
    create_transmission,
    library_dir,
    load_files_from_config,
)


def _create_library_for_template(group, sed_dict, band_dict, cosmo):
    # Split template number off of group
    template, group = group

    # Start with redshift and transmission as None
    redshift = None
    transmission = None

    # Loop over emission line params
    results = []
    for (line_scale, line_fwhm), group1 in group0.groupby(["line_scale", "line_fwhm"]):
        # Get the SED
        sx = sed_dict[int(template)]["x"]
        sy = sed_dict[int(template)]["y"]

        # Add emission lines
        sx, sy = add_emission_lines(sx, sy, line_scale, line_fwhm)

        # Create the transmission matrix
        if redshift is None or transmission is None:
            redshift, transmission = create_transmission(sx, group1, cosmo)

        # Calculate magnitudes
        mags = calc_mags(sx, sy, band_dict, redshift, transmission)

        # Package everything into dataframes
        mags = pd.DataFrame(
            mags,
            columns=band_dict.keys(),
            index=group1.index,
        )
        results.append(pd.concat((group1, mags), axis=1))

    # Combine all results into one dataframe
    results = pd.concat(results, axis=0)

    return results


def main(config_file, force):
    # Check if the file already exists
    file = library_dir / f"{config_file}_library.parquet"
    if file.exists() and not force:
        raise RuntimeError(
            f"Library file {file} already exists. "
            "If you wish to re-create the library, use the flag --force (or -f)."
        )

    # Load the config
    config = Config(library_dir / f"{config_file}.yaml")
    config_grid = config.grid

    # Create function to calculate mags for this config
    sed_dict, band_dict = load_files_from_config(config, 912, 25_000)
    cosmo = jc.Planck15()
    create_library_for_template = lambda group: _create_library_for_template(
        group,
        sed_dict,
        band_dict,
        cosmo,
    )

    # Deploy workers to calculate magnitudes for each template
    with MultiPool() as pool:
        results = pool.map(create_library_for_template, config_grid.groupby("template"))

    # Combine all results into one dataframe
    results = pd.concat(results, axis=0)

    # Save results
    results.to_parquet(file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("library", help="Name of config in libraries/ to use")
    parser.add_argument(
        "-f",
        "--force",
        action="store_true",
        help="Force recreation of libraries that already exist.",
    )

    args = parser.parse_args()
    main(args.library, args.force)
