set -e
git -C /Users/cthoyt/dev/pykeen checkout master
python /Users/cthoyt/dev/collation-benchmark/main.py --top 5
git -C /Users/cthoyt/dev/pykeen checkout negative-sampling-in-data-loader
python /Users/cthoyt/dev/collation-benchmark/main.py --top 5
python /Users/cthoyt/dev/collation-benchmark/compare.py
git commit --all -m "Ran benchmark"
git push
