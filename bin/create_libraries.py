import argparse

import pandas as pd
from schwimmbad import MultiPool

from lumfunc import SED, Bandpasses, Config, library_dir


def main(library, force):
    # Load the config
    lib_path = library_dir / library
    config = Config(lib_path / "config.yaml")
    config_grid = config.grid

    N_full = len(config_grid)
    print(f"The config grid has {N_full} rows")

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
    N_comp = 0 # Track how many rows of the grid are done
    for ID, group in config_grid.groupby("template"):
        # Add the number of rows in this group to the number completed
        N_comp += len(group)

        # Determine the file name for this template
        file = lib_path / f"template{ID}_library.parquet"

        # Skip this template if a library already exists
        if file.exists() and not force:
            print(
                f"Skipping template {ID} because a library already exists at '{file}'"
                f"  [{N_comp / N_full * 100:.1f}% done]"
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
        results = results.set_index(group.index)
        results = pd.concat((group, results), axis=1)

        # Save the library
        print(f"Saving '{file}' ...", end=" ")
        results.to_parquet(file)
        print("Done!", end="  ")

        # Print the fraction of the config grid that we have completed
        print(f"[{N_comp / N_full * 100:.1f}% done]")


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
