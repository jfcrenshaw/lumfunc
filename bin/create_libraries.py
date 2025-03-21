import argparse

import pandas as pd
from schwimmbad import MultiPool

from lumfunc import SED, Bandpasses, Config, library_dir


def main(library, force):
    # Load the config
    lib_path = library_dir / library
    config = Config(lib_path / "config.yaml")
    config_grid = config.grid

    # Load the bandpasses
    bandpasses = Bandpasses(config.filter_files)

    # Define a function to calculate magnitudes for a row in the config grid
    def get_mags_for_row(row):
        row = row[1]
        sed = SED(
            path=config.sed_files[row.template],
            igm_params=dict(
                model=row.igm_model,
                scale=row.igm_scale,
            ),
            dust_params=dict(
                delta=row.dust_delta,
                Av=row.dust_Av,
            ),
            emission_line_params=dict(
                scale=row.line_scale,
                fwhm=row.line_fwhm,
            ),
            redshift=row.redshift,
            bandpasses=bandpasses,
        )
        return sed.mags

    # Loop over templates
    for ID, group in config_grid.groupby("template"):
        # Determine the file name for this template
        file = lib_path / f"template{ID}_library.parquet"

        # Skip this template if a library already exists
        if file.exists() and not force:
            print(
                f"Skipping template {ID} because a library already exists at '{file}'"
            )
            continue

        # Deploy workers to calculate magnitudes
        with MultiPool() as pool:
            results = pool.map(get_mags_for_row, group.iterrows())

        # Pack magnitudes into a dataframe
        results = pd.DataFrame(
            results,
            columns=bandpasses.bands,
        )

        # Add hyperparameters to dataframe
        results = pd.concat((group, results), axis=1)

        # Save the library
        print(f"Saving '{file}' ...", end=" ")
        results.to_parquet(file)
        print("Done!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("library", help="Name of libraries/ subdirectory to use")
    parser.add_argument(
        "-f",
        "--force",
        action="store_true",
        help="Force recreation of libraries that already exist.",
    )

    args = parser.parse_args()
    main(args.library, args.force)
