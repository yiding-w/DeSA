WORK_DIR=/scratch/mrm2vx/Search-R1
LOCAL_DIR=$WORK_DIR/data/nq_hotpotqa_query_generation_sft

## process multiple dataset search format train file
DATA=nq,hotpotqa
python $WORK_DIR/scripts/sft_data_prepare/query_generation_train_merge.py --local_dir $LOCAL_DIR --data_sources $DATA

