# GraphAAAI
Official code for the GraphAAAI (Graph Attentive Adversarial Attack-based Induction) framework.

Please refer to the [framework setup](#framework-setup) section of the README to setup the framework to run the code.

## Reproducing the paper results on an example dataset

This is the official code we used to generate the results presented in the paper. For instance, to reproduce the results for the `Amazon-photo` dataset averaged over five splits, you can use this command:

```bash
for seed in 0 1 2 3 4; do
  CUDA_VISIBLE_DEVICES=$seed ./main.py \
    --graph_dataset amazon-photo \
    --model_dir ~/models/amazon_photo_results_${seed}/ \
    --attention_dim 64 \
    --attention_radius 1 \
    --beta_start 0 \
    --beta_end 0.25 \
    --beta_increment 0.1 \
    --cuda \
    --dropout 0.2 \
    --graph_availability inductive \
    --degree_distance_embed_dim 16 \
    --lr 1e-4 \
    --max_degree 30 \
    --max_neighbors 30 \
    --max_perturbations_per_node 3  \
    --data_partitioning_seed ${seed} \
    --num_attention_heads 32 \
    --output_dim 64 \
    --perturbed_loss_start_epoch 15 \
    --progress \
    --run_type end-to-end \
    --supervised_model_patience 10 \
    --supervised_model_dims 64 \
    --transformer_fc_dim 64 \
    --weight_decay 5e-4 >/dev/null 2>/dev/null &
  sleep 3
done
```

This command assumes that we have five parallel GPUs to train the models. The `--cuda` flag can be removed to run the code on the CPU, and the `CUDA_VISIBLE_DEVICES` environment variable can be adjusted according to the number of available GPUs.

We can now examine these output results and obtain:
```bash
tail ~/models/amazon_photo_results_*/train.log | grep -e 'Accuracy' | sort
```
```
[Test] Accuracy = 0.9589
[Test] Accuracy = 0.9569
[Test] Accuracy = 0.9555
[Test] Accuracy = 0.9582
[Test] Accuracy = 0.9589
[Val] Accuracy = 0.9562
[Val] Accuracy = 0.9562
[Val] Accuracy = 0.9542
[Val] Accuracy = 0.9555
[Val] Accuracy = 0.9569
```
which gives us the reported 0.958 ± 0.003 test accuracy.

## Framework setup

The code is written on top of PyTorch using CUDA 10.2 and Python 3.8.2. We use [Miniconda](https://docs.conda.io/en/latest/miniconda.html) to manage Python packages and provide [this requirements file](requirements.txt) of the exact package versions used to reproduce our results. Interested readers can also contact us directly or raise issues they face on GitHub.
