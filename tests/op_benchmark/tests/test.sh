dlprof --force=true --mode=pytorch --output_path=out_nms --reports=all --formats=json python nms.py --config_id 0 --use_gpu True --repeat 1000 --allow_adaptive_repeat True # nms_cuda
dlprof --force=true --mode=pytorch --output_path=out_roi_align --reports=all --formats=json python roi_align.py --config_id 0 --use_gpu True --repeat 1000 --allow_adaptive_repeat True # roi_align_forward_cuda_kernel
cd ../common/
python performance.py
