#! /bin/bash



if [ "$1" == 1 ] ; then
	run_cmd="OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python a3c.py --name SUDOKU_9_25_EVAL_TEST with main_loop=es_eval_main vlabel_dim=2 clabel_dim=0 state_dim=0 max_iters=2 solver=sharpsat episode_provider=RandomEpisodeProvider episodes_per_batch=48 test_every=10 es_validation_data=./data/sudoku_9_25_validation/ max_step=100000 test_parallelism=2 env_as_process=True sharp_log_actions=False base_model=./saved_models/sudoku/checkpoint-1002"
fi
if [ "$1" == 2 ] ; then
	run_cmd="OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python a3c.py --name SUDOKU_16_105_EVAL_TEST with main_loop=es_eval_main vlabel_dim=2 clabel_dim=0 state_dim=0 max_iters=2 solver=sharpsat episode_provider=RandomEpisodeProvider episodes_per_batch=48 test_every=10 es_validation_data=./data/sudoku_16_105_test/ max_step=100000 test_parallelism=2 env_as_process=True sharp_log_actions=False base_model=./saved_models/sudoku/checkpoint-1002"
fi
if [ "$1" == 3 ] ; then
	run_cmd="OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python a3c.py --name BVEXPR_5_4_8_EVAL_TEST with main_loop=es_eval_main vlabel_dim=0 clabel_dim=0 state_dim=0 max_iters=2 solver=sharpsat episode_provider=RandomEpisodeProvider episodes_per_batch=48 test_every=10 es_validation_data=./data/bvexpr_5_4_8_test max_step=100000 test_parallelism=2 env_as_process=True sharp_log_actions=False base_model=./saved_models/bvexpr/checkpoint-41"
fi
if [ "$1" == 4 ] ; then
	run_cmd="OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python a3c.py --name BVEXPR_5_4_8_EVAL_TEST with main_loop=es_eval_main vlabel_dim=0 clabel_dim=0 state_dim=0 max_iters=2 solver=sharpsat episode_provider=RandomEpisodeProvider episodes_per_batch=48 test_every=10 es_validation_data=./data/bvexpr_7_4_12_test max_step=100000 test_parallelism=2 env_as_process=True sharp_log_actions=False base_model=./saved_models/bvexpr/checkpoint-41"
fi
if [ "$1" == 5 ] ; then
	run_cmd="OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python a3c.py --name CELL49_128_110_EVAL_TEST with main_loop=es_eval_main vlabel_dim=0 clabel_dim=0 state_dim=0 max_iters=2 solver=sharpsat episode_provider=RandomEpisodeProvider episodes_per_batch=48 test_every=10 es_validation_data=./data/cell_49_128_110_test max_step=100000 test_parallelism=2 env_as_process=True sharp_log_actions=False base_model=./saved_models/cell49/checkpoint-581"
fi
if [ "$1" == 6 ] ; then
	run_cmd="OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python a3c.py --name CELL49_192_128_EVAL_TEST with main_loop=es_eval_main vlabel_dim=0 clabel_dim=0 state_dim=0 max_iters=2 solver=sharpsat episode_provider=RandomEpisodeProvider episodes_per_batch=48 test_every=10 es_validation_data=./data/cell_49_192_128_test max_step=100000 test_parallelism=2 env_as_process=True sharp_log_actions=False base_model=./saved_models/cell49/checkpoint-581"
fi
if [ "$1" == 7 ] ; then
	run_cmd="OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python a3c.py --name CELL49_256_200_EVAL_TEST with main_loop=es_eval_main vlabel_dim=0 clabel_dim=0 state_dim=0 max_iters=2 solver=sharpsat episode_provider=RandomEpisodeProvider episodes_per_batch=48 test_every=10 es_validation_data=./data/cell_49_256_200_test max_step=100000 test_parallelism=2 env_as_process=True sharp_log_actions=False base_model=./saved_models/cell49/checkpoint-581"
fi
if [ "$1" == 8 ] ; then
	run_cmd="OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python a3c.py --name CELL35_128_110_EVAL_TEST with main_loop=es_eval_main vlabel_dim=0 clabel_dim=0 state_dim=0 max_iters=2 solver=sharpsat episode_provider=RandomEpisodeProvider episodes_per_batch=48 test_every=10 es_validation_data=./data/cell_35_128_110_test max_step=100000 test_parallelism=2 env_as_process=True sharp_log_actions=False base_model=./saved_models/cell35/checkpoint-981"
fi
if [ "$1" == 9 ] ; then
	run_cmd="OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python a3c.py --name CELL35_192_128_EVAL_TEST with main_loop=es_eval_main vlabel_dim=0 clabel_dim=0 state_dim=0 max_iters=2 solver=sharpsat episode_provider=RandomEpisodeProvider episodes_per_batch=48 test_every=10 es_validation_data=./data/cell_35_192_128_test max_step=100000 test_parallelism=2 env_as_process=True sharp_log_actions=False base_model=./saved_models/cell35/checkpoint-981"
fi
if [ "$1" == 10 ] ; then
	run_cmd="OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python a3c.py --name CELL35_348_280_EVAL_TEST with main_loop=es_eval_main vlabel_dim=0 clabel_dim=0 state_dim=0 max_iters=2 solver=sharpsat episode_provider=RandomEpisodeProvider episodes_per_batch=48 test_every=10 es_validation_data=./data/cell_35_348_280_test max_step=100000 test_parallelism=2 env_as_process=True sharp_log_actions=False base_model=./saved_models/cell35/checkpoint-981"
fi
if [ "$1" == 11 ] ; then
	run_cmd="OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python a3c.py --name CELL9_20_20_EVAL_TEST with main_loop=es_eval_main vlabel_dim=0 clabel_dim=0 state_dim=0 max_iters=2 solver=sharpsat episode_provider=RandomEpisodeProvider episodes_per_batch=48 test_every=10 es_validation_data=./data/cell_9_20_20_test max_step=100000 test_parallelism=2 env_as_process=True sharp_log_actions=False base_model=./saved_models/cell9/checkpoint-1002"
fi
if [ "$1" == 12 ] ; then
	run_cmd="OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python a3c.py --name CELL9_40_40_EVAL_TEST with main_loop=es_eval_main vlabel_dim=0 clabel_dim=0 state_dim=0 max_iters=2 solver=sharpsat episode_provider=RandomEpisodeProvider episodes_per_batch=48 test_every=10 es_validation_data=./data/cell_9_40_40_test max_step=100000 test_parallelism=2 env_as_process=True sharp_log_actions=False base_model=./saved_models/cell9/checkpoint-1002"
fi
if [ "$1" == 13 ] ; then
	run_cmd="OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python a3c.py --name GRID_10_5_EVAL_TEST with main_loop=es_eval_main vlabel_dim=0 clabel_dim=0 state_dim=0 max_iters=2 solver=sharpsat episode_provider=RandomEpisodeProvider episodes_per_batch=48 test_every=10 es_validation_data=./data/grid_10_5_test max_step=100000 test_parallelism=2 env_as_process=True sharp_decode=True sharp_decode_class=GridModel2 sharp_log_actions=False sharp_decode_size=10 sharp_decoded_emb_dim=1 base_model=./saved_models/grid/checkpoint-521"
fi
if [ "$1" == 14 ] ; then
	run_cmd="OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python a3c.py --name GRID_10_12_EVAL_TEST with main_loop=es_eval_main vlabel_dim=0 clabel_dim=0 state_dim=0 max_iters=2 solver=sharpsat episode_provider=RandomEpisodeProvider episodes_per_batch=48 test_every=10 es_validation_data=./data/grid_10_12_test max_step=100000 test_parallelism=2 env_as_process=True sharp_decode=True sharp_decode_class=GridModel2 sharp_log_actions=False sharp_decode_size=10 sharp_decoded_emb_dim=1 base_model=./saved_models/grid/checkpoint-521"
fi
if [ "$1" == 15 ] ; then
	run_cmd="OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python a3c.py --name GRID_10_14_EVAL_TEST with main_loop=es_eval_main vlabel_dim=0 clabel_dim=0 state_dim=0 max_iters=2 solver=sharpsat episode_provider=RandomEpisodeProvider episodes_per_batch=48 test_every=10 es_validation_data=./data/grid_10_14_test max_step=100000 test_parallelism=2 env_as_process=True sharp_decode=True sharp_decode_class=GridModel2 sharp_log_actions=False sharp_decode_size=10 sharp_decoded_emb_dim=1 base_model=./saved_models/grid/checkpoint-521"
fi
if [ "$1" == 15 ] ; then
	run_cmd="OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python a3c.py --name GRID_12_14_EVAL_TEST with main_loop=es_eval_main vlabel_dim=0 clabel_dim=0 state_dim=0 max_iters=2 solver=sharpsat episode_provider=RandomEpisodeProvider episodes_per_batch=48 test_every=10 es_validation_data=./data/grid_12_14_test max_step=100000 test_parallelism=2 env_as_process=True sharp_decode=True sharp_decode_class=GridModel2 sharp_log_actions=False sharp_decode_size=10 sharp_decoded_emb_dim=1 base_model=./saved_models/grid/checkpoint-521"
fi

if [ "$2" == vanilla ] ; then
	run_cmd="${run_cmd} es_vanilla_policy=True"
fi
echo $run_cmd
eval $run_cmd
./clean_ray.sh
