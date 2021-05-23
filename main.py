# -*- coding: utf-8 -*-

"""Main script."""

import getpass
import logging
import pathlib
from itertools import chain, product
from typing import Iterable, Optional

import click
import math
import pandas as pd
import seaborn as sns
from docdata import get_docdata
from more_click import force_option, verbose_option
from torch.optim import Adam
from torch.utils.benchmark import Timer
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

import pykeen
from pykeen import losses
from pykeen.datasets import Dataset, datasets as dataset_dict, get_dataset
from pykeen.losses import loss_resolver
from pykeen.models import model_resolver
from pykeen.sampling import negative_sampler_resolver
from pykeen.sampling.filtering import BloomFilterer, filterer_resolver
from pykeen.training import LCWATrainingLoop, NonFiniteLossError, SLCWATrainingLoop
from pykeen.utils import resolve_device

logger = logging.getLogger(__name__)

VERSION = pykeen.get_version()

HERE = pathlib.Path(__file__).resolve().parent
DATA = HERE.joinpath("data", getpass.getuser())
DEFAULT_DIRECTORY = DATA.joinpath(pykeen.get_git_branch(), pykeen.get_git_hash())
DEFAULT_DIRECTORY.mkdir(exist_ok=True, parents=True)

#: Columns in each dataset-specific file
COLUMNS = ["trainer", "loss", "sampler", "filterer", "num_negs_per_pos", "time", "frequency"]


@click.command()
@click.option("--epochs", type=int, default=3, show_default=True)
@click.option("--dataset")
@click.option("--top", type=int)
@verbose_option
@force_option
def main(dataset: Optional[str], epochs: int, top: Optional[int], force: bool) -> None:
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

    for dataset_instance in _iterate_datasets(dataset, top=top):
        with logging_redirect_tqdm():
            df = _generate(
                dataset=dataset_instance, device=device, epochs=epochs, force=force
            )
        df_columns = df.columns
        df["dataset"] = dataset_instance.get_normalized_name()
        df = df[["dataset", *df_columns]]
        dfs.append(df)

    sdf = pd.concat(dfs)
    sdf.to_csv(DEFAULT_DIRECTORY.joinpath("results.tsv.gz"), sep="\t", index=False)

    g = plot(sdf)
    g.fig.savefig(DEFAULT_DIRECTORY.joinpath("output.svg"))
    g.fig.savefig(DEFAULT_DIRECTORY.joinpath("output.png"), dpi=300)


def _generate(*, dataset: Dataset, epochs, device, force: bool = False) -> pd.DataFrame:
    path = DEFAULT_DIRECTORY.joinpath(dataset.get_normalized_name()).with_suffix(".tsv")
    if path.is_file() and not force:
        return pd.read_csv(path, sep="\t")

    data = []

    it = _keys(dataset=dataset)
    for i, (
        loop_cls,
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
            loop=loop_cls.get_normalized_name(),
            loss=loss_resolver.normalize_cls(loss_cls),
            sampler=negative_sampler_cls and negative_sampler_cls.get_normalized_name(),
            filterer=filterer_cls and filterer_resolver.normalize_cls(filterer_cls),
            num_negs_per_pos=num_negs_per_pos,
        )
        if loop_cls is SLCWATrainingLoop:
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
        else:
            trainer = LCWATrainingLoop(
                model=model,
                optimizer=optimizer,
                triples_factory=dataset.training,
                automatic_memory_optimization=False,
            )

        timer = Timer(
            stmt="""\
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
                loop_cls.get_normalized_name(),
                loss_resolver.normalize_cls(loss_cls),
                negative_sampler_cls and negative_sampler_cls.get_normalized_name(),
                filterer_cls and filterer_resolver.normalize_cls(filterer_cls),
                num_negs_per_pos,
                t,
                epochs / t,
            )
            for t in measurement.raw_times
        )

    df = pd.DataFrame(data, columns=COLUMNS)
    df.to_csv(path, sep="\t", index=False)
    return df


def _keys(dataset: Dataset):
    num_negs_per_pos_values = [10 ** i for i in range(3)]
    losses_list = [
        # losses.NSSALoss,
        losses.SoftplusLoss,
        # losses.MarginRankingLoss,
    ]
    slcwa_keys = (
        [SLCWATrainingLoop],
        losses_list,
        list(negative_sampler_resolver),
        [None, BloomFilterer],
        num_negs_per_pos_values,
    )
    lcwa_keys = ([LCWATrainingLoop], losses_list, [None], [None], [0])
    all_keys = (lcwa_keys, slcwa_keys)
    it = tqdm(
        chain.from_iterable(product(*keys) for keys in all_keys),
        desc=dataset.get_normalized_name(),
        total=sum(math.prod(len(k) for k in keys) for keys in all_keys),
    )
    return it


def _make_label(trainer, sampler, filterer):
    if trainer == "lcwa":
        return trainer
    if not filterer or pd.isnull(filterer) or pd.isna(filterer):
        filterer = "unfiltered"
    return f"{trainer}/{sampler}/{filterer}"


def _add_label(df, key):
    df[key] = [
        _make_label(trainer, sampler, filterer)
        for trainer, sampler, filterer in df[["trainer", "sampler", "filterer"]].values
    ]


def plot(df: pd.DataFrame, row=None):
    hue_key = "hue"
    _add_label(df, hue_key)

    g = sns.relplot(
        data=df,
        x="num_negs_per_pos",
        y="time",
        hue=hue_key,
        kind="line",
        col="dataset",
        row=row,
        height=3.5,
        # ci=100,
        # estimator=numpy.median,
    )
    g.set(
        xscale="log",
        yscale="log",
        #     xlabel="Batch Size",
        #     ylabel="Seconds Per Batch",
    )
    g.tight_layout()
    return g


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
