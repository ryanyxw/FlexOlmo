
# Experiments in the paper

## Public model (anchor)

### Base training

Schedule set to 5T tokens, training stopped at 1T tokens

```bash
torchrun --nproc-per-node=8 ./src/scripts/train/OLMo2-7B-train.py OLMo2-7B-public-mix \
    --dataset.mix_base_dir=/path/to/tokenized/data \
    --train_module.float8_config.enabled=true \
    --trainer.duration.value=5_000_000_000_000 \
    --trainer.duration.unit=tokens \
    --trainer.hard_stop.value=1_000_000_000_000 \
    --trainer.hard_stop.unit=tokens
```

### Anneal for 50B tokens

```bash
CHECKPOINT=/path/to/checkpoints/OLMo2-7B-public-mix/step238419
torchrun --nproc-per-node=8 ./src/scripts/train/OLMo2-7B-anneal.py OLMo2-7B-anneal-public-mix ${CHECKPOINT}
    --dataset.mix_base_dir=/path/to/tokenized/data \
    --train_module.float8_config.enabled=true \
    --trainer.duration.value=50_000_000_000\
    --trainer.duration.unit=tokens
```

#### Unshard model

```bash
MODEL_PATH=/path/to/checkpoints/OLMo2-7B-anneal-public-mix/step11921
python src/scripts/utils/unshard.py -i ${MODEL_PATH} -o ${MODEL_PATH}-unsharded
```


## Train 2x7B expert models

### Upcycle dense 7B public model into 2x7B MoE

Note: This only needs to happen once, and then, all experts can use this.

```bash
MODEL_PATH=/path/to/checkpoints/OLMo2-7B-anneal-public-mix/step11921
python src/scripts/upcycle/dense_to_expert_moe.py -m ${MODEL_PATH}-unsharded ${MODEL_PATH}-unsharded -t /path/to/checkpoints/olmoe-2x7b-public-public
```

With domain embeddings for router for each individual expert (e.g. news)

```bash
MODEL_PATH=/path/to/checkpoints/OLMo2-7B-anneal-public-mix/step11921
python src/scripts/upcycle/dense_to_expert_moe.py -m ${MODEL_PATH}-unsharded ${MODEL_PATH}-unsharded -e /path/to/public/embeds /path/to/news/embeds -t /path/to/checkpoints/olmoe-2x7b-public-news
```


### Expert training

```bash
MODEL_PATH=/path/to/checkpoints/olmoe-2x7b-public-math  # upcycled from previous step
torchrun --nproc-per-node=8 src/scripts/train/OLMoE-2x7B-anneal.py olmoe-2x7B-news_top2_grit_learnbias \
    --trainer.callbacks.profiler.enabled=true \
    --dataset.mix_base_dir=/path/to/tokenized/data \
    --dataset.mix=news \
    --trainer.max_duration.value=50_000_000_000 \
    --trainer.max_duration.unit=tokens \
    --trainer.load_path=${MODEL_PATH} \
    --model.block.feed_forward_moe.router.top_k=2 \
    --train_module.rank_microbatch_size=4096 \
    --train_module.scheduler.warmup_steps=2000 \
    --train_module.optim.lr=9e-4
```

## Combine experts into FlexOlmo

```bash
PUBLIC_EXPERT=/path/to/public/expert
EXPERT_1=/path/to/expert1
EXPERT_2=/path/to/expert2
.
.
python src/scripts/upcycle/merge_experts_to_flexolmo.py -m ${PUBLIC_EXPERT}-unsharded -m ${EXPERT_1}-unsharded -m ${EXPERT_2}-unsharded [..OTHER EXPERTS]
```

### [Optional] Additional router training

```bash
torchrun ...
```

## Evaluation

### Convert models to HF

```bash
python ...
```

### Local evaluation, without beaker

TODO
```
```

## Data extraction