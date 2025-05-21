CUDA_VISIBLE_DEVICES=1 python merge_lora_weights_and_save_hf_model_modi.py \
  --version="/path/to/GLOVER++" \
  --weight="/path/to/pytorch_model.bin" \
  --save_path="/path/to/glover++/save/dir" \
  --model_arch="glover++"