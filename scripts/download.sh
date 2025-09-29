
save_path=/path/to/your/cache

python download.py --savepath $savepath

cat $save_path/part_* > e5_Flat.index
