set -e
git -C ../pykeen checkout master
python main.py --top 4
git -C ../pykeen checkout negative-sampling-in-data-loader
python main.py --top 4
python compare.py
#git commit --all -m "Ran benchmark"
#git push
