
cd DCNet

pip install -r requirements.txt


python -u search_best_main.py --config config/MSL.conf


# 第一批：bs=32 的5个
CUDA_VISIBLE_DEVICES=0 python search_best_main.py --config config/MSL_grid/MSL_bs32_d16.conf &
CUDA_VISIBLE_DEVICES=0 python search_best_main.py --config config/MSL_grid/MSL_bs32_d32.conf &
CUDA_VISIBLE_DEVICES=0 python search_best_main.py --config config/MSL_grid/MSL_bs32_d64.conf &
CUDA_VISIBLE_DEVICES=0 python search_best_main.py --config config/MSL_grid/MSL_bs32_d128.conf &
CUDA_VISIBLE_DEVICES=0 python search_best_main.py --config config/MSL_grid/MSL_bs32_d256.conf &
wait

# 第二批：bs=64 的5个
CUDA_VISIBLE_DEVICES=0 python search_best_main.py --config config/MSL_grid/MSL_bs64_d16.conf &
CUDA_VISIBLE_DEVICES=0 python search_best_main.py --config config/MSL_grid/MSL_bs64_d32.conf &
CUDA_VISIBLE_DEVICES=0 python search_best_main.py --config config/MSL_grid/MSL_bs64_d64.conf &
CUDA_VISIBLE_DEVICES=0 python search_best_main.py --config config/MSL_grid/MSL_bs64_d128.conf &
CUDA_VISIBLE_DEVICES=0 python search_best_main.py --config config/MSL_grid/MSL_bs64_d256.conf &
wait

# 第三批：bs=128 的5个
CUDA_VISIBLE_DEVICES=0 python search_best_main.py --config config/MSL_grid/MSL_bs128_d16.conf &
CUDA_VISIBLE_DEVICES=0 python search_best_main.py --config config/MSL_grid/MSL_bs128_d32.conf &
CUDA_VISIBLE_DEVICES=0 python search_best_main.py --config config/MSL_grid/MSL_bs128_d64.conf &
CUDA_VISIBLE_DEVICES=0 python search_best_main.py --config config/MSL_grid/MSL_bs128_d128.conf &
CUDA_VISIBLE_DEVICES=0 python search_best_main.py --config config/MSL_grid/MSL_bs128_d256.conf &
wait

# 第四批：bs=256 的5个
CUDA_VISIBLE_DEVICES=0 python search_best_main.py --config config/MSL_grid/MSL_bs256_d16.conf &
CUDA_VISIBLE_DEVICES=0 python search_best_main.py --config config/MSL_grid/MSL_bs256_d32.conf &
CUDA_VISIBLE_DEVICES=0 python search_best_main.py --config config/MSL_grid/MSL_bs256_d64.conf &
CUDA_VISIBLE_DEVICES=0 python search_best_main.py --config config/MSL_grid/MSL_bs256_d128.conf &
CUDA_VISIBLE_DEVICES=0 python search_best_main.py --config config/MSL_grid/MSL_bs256_d256.conf &
wait

# 第五批：bs=512 的5个
CUDA_VISIBLE_DEVICES=0 python search_best_main.py --config config/MSL_grid/MSL_bs512_d16.conf &
CUDA_VISIBLE_DEVICES=0 python search_best_main.py --config config/MSL_grid/MSL_bs512_d32.conf &
CUDA_VISIBLE_DEVICES=0 python search_best_main.py --config config/MSL_grid/MSL_bs512_d64.conf &
CUDA_VISIBLE_DEVICES=0 python search_best_main.py --config config/MSL_grid/MSL_bs512_d128.conf &
CUDA_VISIBLE_DEVICES=0 python search_best_main.py --config config/MSL_grid/MSL_bs512_d256.conf &

# 第六批：bs=1024 的5个
CUDA_VISIBLE_DEVICES=0 python search_best_main.py --config config/MSL_grid/MSL_bs1024_d16.conf &
CUDA_VISIBLE_DEVICES=0 python search_best_main.py --config config/MSL_grid/MSL_bs1024_d32.conf &
CUDA_VISIBLE_DEVICES=0 python search_best_main.py --config config/MSL_grid/MSL_bs1024_d64.conf &
CUDA_VISIBLE_DEVICES=0 python search_best_main.py --config config/MSL_grid/MSL_bs1024_d128.conf &
CUDA_VISIBLE_DEVICES=0 python search_best_main.py --config config/MSL_grid/MSL_bs1024_d256.conf &
wait

export CUBLAS_WORKSPACE_CONFIG=:4096:8
