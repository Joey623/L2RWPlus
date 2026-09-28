"""Prepare a dataset from its official source files."""

import argparse


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", required=True, choices=["sysu", "regdb", "llcm", "vcm", "bupt"])
    p.add_argument("--data-root", default="data")
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--overwrite", action="store_true")
    group = p.add_mutually_exclusive_group()
    group.add_argument("--index-only", action="store_true")
    group.add_argument("--lmdb-only", action="store_true")
    a = p.parse_args(argv)
    base = ["--dataset", a.dataset, "--data-root", a.data_root]
    if a.overwrite:
        base.append("--overwrite")
    if not a.index_only:
        from l2rwplus.data.preparation.lmdb import main as build_lmdb

        build_lmdb(base + ["--workers", str(a.workers)])
    if not a.lmdb_only:
        from l2rwplus.data.preparation.indexes import main as build_indexes

        build_indexes(base)


if __name__ == "__main__":
    main()
