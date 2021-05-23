# -*- coding: utf-8 -*-

"""Main script."""

import getpass
import logging
import pathlib
from itertools import product
from typing import Iterable, Optional

import click
import math
import pandas as pd
import seaborn as sns
from docdata import get_docdata
from more_click import verbose_option
from torch.optim import Adam
from torch.utils.benchmark import Timer
from tqdm import tqdm

import pykeen
from pykeen import losses
from pykeen.datasets import Dataset, datasets as dataset_dict, get_dataset
from pykeen.losses import loss_resolver
from pykeen.models import model_resolver
from pykeen.sampling import negative_sampler_resolver
from pykeen.sampling.filtering import BloomFilterer, filterer_resolver
from pykeen.training import NonFiniteLossError, SLCWATrainingLoop
from pykeen.utils import resolve_device

logger = logging.getLogger(__name__)

USER = getpass.getuser()
VERSION = pykeen.get_version()
GIT_HASH = pykeen.get_git_hash()
GIT_BRANCH = pykeen.get_git_branch()

HERE = pathlib.Path(__file__).resolve().parent
DEFAULT_DIRECTORY = HERE.joinpath("data", USER, GIT_HASH)
DEFAULT_DIRECTORY.mkdir(exist_ok=True, parents=True)


@click.command()
@click.option("--epochs", type=int, default=20, show_default=True)
@click.option("--dataset")
@verbose_option
def main(dataset: Optional[str], epochs: int) -> None:
    """Run the benchmark.

    Things to measure:
    - time
    - memory consumption (secondary)

    Things to benchmark across datasets/models

    1. LCWA
    2. sLCWA
       - filtered/unfiltered
       - basic/bernoulli/pseudotyped
    """
    dfs = []
    device = resolve_device()

    for dataset_instance in _iterate_datasets(dataset, top=1):
        df = _generate(dataset=dataset_instance, device=device, epochs=epochs)
        df_columns = df.columns
        df["dataset"] = dataset_instance.get_normalized_name()
        df = df[["dataset", *df_columns]]
        dfs.append(df)

    sdf = pd.concat(dfs)
    _plot(sdf)


COLUMNS = ["loss", "sampler", "filterer", "num_negs_per_pos", "time"]


def _generate(*, dataset: Dataset, epochs, device, force: bool = False) -> pd.DataFrame:
    path = DEFAULT_DIRECTORY.joinpath(dataset.get_normalized_name()).with_suffix(".tsv")
    if path.is_file() and not force:
        return pd.read_csv(path, sep="\t")

    data = []
    num_negs_per_pos_values = [10 ** i for i in range(3)]
    keys = (
        [
            # losses.NSSALoss,
            losses.SoftplusLoss,
            # losses.MarginRankingLoss,
        ],
        list(negative_sampler_resolver),
        [None, BloomFilterer],
        num_negs_per_pos_values,
    )
    it = tqdm(
        product(*keys),
        desc=dataset.get_normalized_name(),
        total=math.prod(len(k) for k in keys),
    )
    for i, (
        loss_cls,
        negative_sampler_cls,
        filterer_cls,
        num_negs_per_pos,
    ) in enumerate(it):
        model = model_resolver.make(
            "TransE",
            triples_factory=dataset.training,
            preferred_device=device,
            random_seed=i,
            loss=loss_resolver.make(loss_cls),
        )
        optimizer = Adam(model.parameters())

        it.set_postfix(
            loss=loss_resolver.normalize_cls(loss_cls),
            sampler=negative_sampler_cls.get_normalized_name(),
            filterer=(
                filterer_resolver.normalize_cls(filterer_cls)
                if filterer_cls
                else "none"
            ),
            num_negs_per_pos=num_negs_per_pos,
        )
        negative_sampler = negative_sampler_resolver.make(
            negative_sampler_cls,
            triples_factory=dataset.training,
            num_negs_per_pos=num_negs_per_pos,
            filterer=filterer_cls,
        )
        trainer = SLCWATrainingLoop(
            model=model,
            optimizer=optimizer,
            triples_factory=dataset.training,
            negative_sampler=negative_sampler,
            automatic_memory_optimization=False,
        )
        timer = Timer(
            stmt="""
                    trainer.train(
                        triples_factory=triples_factory, 
                        num_epochs=num_epochs, 
                        use_tqdm=False,
                        batch_size=batch_size,
                    )
                """,
            globals=dict(
                trainer=trainer,
                triples_factory=dataset.training,
                num_epochs=epochs,
                batch_size=512,
            ),
        )
        try:
            measurement = timer.blocked_autorange()
        except NonFiniteLossError as e:
            tqdm.write(f"error: {e}")
            continue

        data.extend(
            (
                loss_resolver.normalize_cls(loss_cls),
                negative_sampler_cls.get_normalized_name(),
                (
                    filterer_resolver.normalize_cls(filterer_cls)
                    if filterer_cls
                    else "none"
                ),
                num_negs_per_pos,
                t,
            )
            for t in measurement.raw_times
        )

    df = pd.DataFrame(data, columns=COLUMNS)
    df.to_csv(path, sep="\t", index=False)
    return df


def _plot(df: pd.DataFrame):
    g = sns.relplot(
        data=df,
        x="num_negs_per_pos",
        y="time",
        hue="sampler",
        kind="line",
        row="dataset",
        col="filterer",
        height=3.5,
        # ci=100,
        # estimator=numpy.median,
    )
    g.set(
        xscale="log",
        #     xlabel="Batch Size",
        #     ylabel="Seconds Per Batch",
    )
    g.tight_layout()
    g.fig.savefig(DEFAULT_DIRECTORY.joinpath("output.svg"))


def _iterate_datasets(dataset: Optional[str], top=None) -> Iterable[Dataset]:
    if dataset:
        _dataset_list = [dataset]
    else:
        _dataset_list = sorted(dataset_dict, key=_triples)
    if top:
        _dataset_list = _dataset_list[:top]
    it = tqdm(_dataset_list, desc="Dataset")
    for dataset in it:
        it.set_postfix(dataset=dataset)
        yield get_dataset(dataset=dataset)


def _triples(d: str) -> int:
    return get_docdata(dataset_dict[d])["statistics"]["triples"]


if __name__ == "__main__":
    main()
