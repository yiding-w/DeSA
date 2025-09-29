WORK_DIR=
LOCAL_DIR=$WORK_DIR/data/hotpotqa

## process multiple dataset search format train file
DATA=nq,hotpotqa
# DATA=hotpotqa
python $WORK_DIR/scripts/data_process/qa_search_train_merge.py --local_dir $LOCAL_DIR --data_sources $DATA

## process multiple dataset search format test file
DATA=nq,triviaqa,popqa,hotpotqa,2wikimultihopqa,musique,bamboogle
# DATA=hotpotqa
python $WORK_DIR/scripts/data_process/qa_search_test_merge.py --local_dir $LOCAL_DIR --data_sources $DATA
