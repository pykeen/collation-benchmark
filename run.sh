git -C /Users/cthoyt/dev/pykeen checkout master
python /Users/cthoyt/dev/collation-benchmark/main.py --top 4
git -C /Users/cthoyt/dev/pykeen checkout negative-sampling-in-data-loader
python /Users/cthoyt/dev/collation-benchmark/main.py --top 4
python /Users/cthoyt/dev/collation-benchmark/compare.py
