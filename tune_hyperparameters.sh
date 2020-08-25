#!/bin/bash

# Our machine had 8 Tesla 32GB GPUs. Assuming ~2GB per model based on trial runs, we can do 120 models over 8 GPUs
# with some buffer to spare for misc use.
max_running_jobs=120
num_gpus=8
datasets=(cora citeseer pubmed ppi cora-full amazon-photo amazon-computers coauthor-cs coauthor-physics reddit)
hdims=(64 128 256 512)
attention_radii=(1 2)
attention_dims=(64 256)
attention_heads=(8 16 32)
lrs=(1e-4 1e-5)
dropouts=(0 0.2 0.4 0.6)

NO_PERTURBATIONS=10000
perturbed_loss_start_epochs=(5 10 15 20 $NO_PERTURBATIONS)

beta_ends=(0.25 0.5 0.75)
beta_increments=(0.05 0.1 0.25)
max_perturbations=(1 3 5)

trap "exit" INT TERM ERR
trap "kill 0" EXIT

function get_active_jobs() {
  echo $(jobs | grep 'Running' | wc -l)
}

function wait_for_free() {
  while true; do
    active_jobs=$(get_active_jobs)
    if [ "$active_jobs" -lt "$max_running_jobs" ]; then
      return
    else
      sleep 1
    fi
  done
}

function wait_for_finish() {
  previous_active=$(get_active_jobs)
  while true; do
    active_jobs=$(get_active_jobs)
    if [ "$active_jobs" -eq "0" ]; then
      echo "[Active jobs ${active_jobs}]"
      return
    elif [ "$active_jobs" -lt "$previous_active" ]; then
      echo "[Active jobs ${active_jobs}]"
      previous_active=$active_jobs
    fi
    sleep 1
  done
}

function get_next_gpu() {
  dataset_count=$1
  gpu_offset=$(($dataset_count % $num_gpus))
  echo $gpu_offset
}

tuning_date=$(date +%Y-%m-%d_%I.%M%p)
for dataset in "${datasets[@]}"; do
  dataset_count=0
  loss_fn='cross_entropy'
  if [ "$dataset" == "ppi" ]; then
    loss_fn='binary_cross_entropy'
  fi
  for hdim in "${hdims[@]}"; do
    for attention_radius in "${attention_radii[@]}"; do
      for attention_dim in "${attention_dims[@]}"; do
        for num_attention_heads in "${attention_heads[@]}"; do
          for lr in "${lrs[@]}"; do
            for dropout in "${dropouts[@]}"; do
              for perturbed_loss_start_epoch in "${perturbed_loss_start_epochs[@]}"; do
                if [ "$perturbed_loss_start_epoch" -eq "$NO_PERTURBATIONS" ]; then
                  this_beta_ends=(0)
                  this_beta_increments=(0)
                  this_max_perturbations=(0)
                else
                  this_beta_ends=( "${beta_ends[@]}" )
                  this_beta_increments=( "${beta_increments[@]}" )
                  this_max_perturbations=( "${max_perturbations[@]}" )
                fi
                for beta_end in "${this_beta_ends[@]}"; do
                  for beta_increment in "${this_beta_increments[@]}"; do
                    for max_perturbations_per_node in "${this_max_perturbations[@]}"; do
                      wait_for_free
                      gpu_to_use=$(get_next_gpu $dataset_count)
                      dataset_count=$(($dataset_count + 1))
                      CUDA_VISIBLE_DEVICES=$gpu_to_use ./main.py \
                          --graph_dataset $dataset \
                          --model_dir ~/graphaaai_tuning_${tuning_date}/${dataset}_${dataset_count}/ \
                          --attention_dim $attention_dim \
                          --attention_radius $attention_radius \
                          --beta_start 0 \
                          --beta_end $beta_end \
                          --beta_increment $beta_increment \
                          --cuda \
                          --dropout $dropout \
                          --graph_availability inductive \
                          --loss_fn $loss_fn \
                          --lr $lr \
                          --max_degree 30 \
                          --max_neighbors 50 \
                          --max_perturbations_per_node $max_perturbations_per_node \
                          --num_attention_heads $num_attention_heads \
                          --output_dim $hdim \
                          --perturbed_loss_start_epoch $perturbed_loss_start_epoch \
                          --run_type end-to-end \
                          --supervised_model_patience 10 \
                          --supervised_model_dims $hdim \
                          --transformer_fc_dim $hdim \
                          --weight_decay 5e-4 >/dev/null &
                      active_jobs=$(get_active_jobs)
                      echo "[Active jobs ${active_jobs}] Reached $dataset $dataset_count"
                      sleep 2
                    done
                  done
                done
              done
            done
          done
        done
      done
    done
  done
done

# Use this as opposed to simple `wait` so we can see progress.
wait_for_finish
