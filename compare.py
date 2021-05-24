import pathlib

import click
import pandas as pd
import seaborn as sns

from main import DATA


def _get_latest(directory: str) -> pathlib.Path:
    d = DATA.joinpath(directory)
    return next((sd for sd in d.iterdir() if sd.is_dir()))


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

    group_key = "row"
    df[group_key] = [
        f'{sampler or ""}/{filterer or ""}'
        for sampler, filterer in df[["sampler", "filterer"]].values
    ]
    g = sns.relplot(
        data=df[df.trainer == 'slcwa'],
        x="num_negs_per_pos",
        y="frequency",
        hue=branch_key,
        kind="line",
        col="dataset",
        height=3,
        row=group_key,
        markers=True,
        aspect=1.3,
        # ci=100,
        # estimator=numpy.median,
    )
    g.set(
        xscale="log",
        yscale="log",
        #     xlabel="Batch Size",
        ylabel="Epochs per Second",
    )
    g.set_titles("{col_name}: {row_name}")
    g.tight_layout()
    g.fig.savefig(DATA.joinpath("comparison.svg"))
    g.fig.savefig(DATA.joinpath("comparison.png"), dpi=300)


if __name__ == "__main__":
    main()
