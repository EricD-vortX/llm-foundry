cd scripts

# Convert C4 dataset to StreamingDataset format
python data_prep/convert_dataset_hf.py \
  --dataset c4 --data_subset en \
  --out_root my-copy-c4 --splits train_small val_small \
  --concat_tokens 2048 --tokenizer EleutherAI/gpt-neox-20b --eos_text '<|endoftext|>'

# Train an MPT-125m model for 10 batches
composer train/train.py \
  train/yamls/pretrain/mpt-125m.yaml \
  data_local=my-copy-c4 \
  train_loader.dataset.split=train_small \
  eval_loader.dataset.split=val_small \
  max_duration=10ba \
  eval_interval=0 \
  save_folder=mpt-125m

# Convert the model to HuggingFace format
python inference/convert_composer_to_hf.py \
  --composer_path mpt-125m/ep0-ba10-rank0.pt \
  --hf_output_path mpt-125m-hf \
  --output_precision bf16 \
  # --hf_repo_for_upload user-org/repo-name

# Evaluate the model on a subset of tasks
python eval/eval.py \
  eval/yamls/hf_eval.yaml \
  icl_tasks=eval/yamls/copa.yaml \
  model_name_or_path=mpt-125m-hf

# Generate responses to prompts
python inference/hf_generate.py \
  --name_or_path mpt-125m-hf \
  --max_new_tokens 256 \
  --prompts \
    "The answer to life, the universe, and happiness is" \
    "Here's a quick recipe for baking chocolate chip cookies: Start by"