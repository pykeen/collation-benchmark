import pathlib

import click
import pandas as pd

from main import DATA, plot


def _get_latest(directory: str) -> pathlib.Path:
    d = DATA.joinpath(directory)
    return list(d.iterdir())[0]


MASTER_BRANCH = "master"
NEW_BRANCH = "negative-sampling-in-data-loader"


@click.command()
@click.option(
    "--master", type=pathlib.Path, default=_get_latest(MASTER_BRANCH), show_default=True
)
@click.option(
    "--new", type=pathlib.Path, default=_get_latest(NEW_BRANCH), show_default=True
)
def main(master: pathlib.Path, new: pathlib.Path):
    branch_key = "branch"
    left_df = pd.read_csv(
        DATA.joinpath(MASTER_BRANCH, master, "results.tsv.gz"), sep="\t"
    )
    left_df[branch_key] = MASTER_BRANCH
    right_df = pd.read_csv(DATA.joinpath(NEW_BRANCH, new, "results.tsv.gz"), sep="\t")
    right_df[branch_key] = NEW_BRANCH
    df = pd.concat([left_df, right_df])

    g = plot(df, row=branch_key)
    g.fig.savefig(DATA.joinpath("comparison.svg"))
    g.fig.savefig(DATA.joinpath("comparison.png"), dpi=300)


if __name__ == "__main__":
    main()
