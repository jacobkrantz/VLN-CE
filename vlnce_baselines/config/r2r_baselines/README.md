# Beyond the Nav-Graph: Vision-and-Language Navigation in Continuous Environments

Experiments in this paper used the R2R data. Each experiment can be recreated following the configs below:

| Model             | val_seen SPL | val_unseen SPL | Config |
|-------------------|--------------|----------------|--------|
| Seq2Seq           | 0.24         | 0.18           | [seq2seq.yaml](seq2seq.yaml) |
| Seq2Seq_PM        | 0.21         | 0.15           | [seq2seq_pm.yaml](seq2seq_pm.yaml) |
| Seq2Seq_DA        | 0.32         | 0.23           | [seq2seq_da.yaml](seq2seq_da.yaml) |
| Seq2Seq_Aug       | 0.25         | 0.17           | [seq2seq_aug.yaml](seq2seq_aug.yaml)  ⟶ [seq2seq_aug_tune.yaml](seq2seq_aug_tune.yaml) |
| Seq2Seq_PM_DA_Aug | 0.31         | 0.22           | [seq2seq_pm_aug.yaml](seq2seq_pm_aug.yaml)  ⟶ [seq2seq_pm_da_aug_tune.yaml](seq2seq_pm_da_aug_tune.yaml) |
| CMA               | 0.25         | 0.22           | [cma.yaml](cma.yaml) |
| CMA_PM            | 0.26         | 0.19           | [cma_pm.yaml](cma_pm.yaml) |
| CMA_DA            | 0.31         | 0.25           | [cma_da.yaml](cma_da.yaml) |
| CMA_Aug           | 0.24         | 0.19           | [cma_aug.yaml](cma_aug.yaml)  ⟶ [cma_aug_tune.yaml](cma_aug_tune.yaml) |
| **CMA_PM_DA_Aug** | **0.35**     | **0.30**       | [cma_pm_aug.yaml](cma_pm_aug.yaml)  ⟶ [cma_pm_da_aug_tune.yaml](cma_pm_da_aug_tune.yaml) |
| CMA_PM_Aug        | 0.25         | 0.22           | [cma_pm_aug.yaml](cma_pm_aug.yaml)  ⟶ [cma_pm_aug_tune.yaml](cma_pm_aug_tune.yaml) |
| CMA_DA_Aug        | 0.33         | 0.26           | [cma_aug.yaml](cma_aug.yaml)  ⟶ [cma_da_aug_tune.yaml](cma_da_aug_tune.yaml) |

|         |  Legend                                                                                          |
|---------|--------------------------------------------------------------------------------------------------|
| Seq2Seq | Sequence-to-Sequence baseline model                                                              |
| CMA     | Cross-Modal Attention model                                                                      |
| PM      | [Progress monitor](https://github.com/chihyaoma/selfmonitoring-agent)                            |
| DA      | DAgger training (otherwise teacher forcing)                                                      |
| Aug     | Uses the [EnvDrop](https://github.com/airsplay/R2R-EnvDrop) episodes to augment the training set |
| ⟶       | Use the config on the left to train the model. Evaluate each checkpoint on `val_unseen`. The best checkpoint (according to `val_unseen` SPL) is then fine-tuned using the config on the right. Make sure to update the field `IL.ckpt_to_load` before fine-tuning. |

## Pretrained Models

We provide pretrained models for our best Seq2Seq model [Seq2Seq_DA](https://drive.google.com/open?id=14y7dXkAEwB_q81cDCow8JNKD2aAAPxbW) and Cross-Modal Attention model [CMA_PM_DA_Aug](https://drive.google.com/open?id=1xIxh5eUkjGzSL_3AwBqDQlNXjkFrpcg4). These models are hosted on Google Drive and can be downloaded as such:

```bash
# CMA_PM_DA_Aug (141MB)
gdown https://drive.google.com/uc?id=1xIxh5eUkjGzSL_3AwBqDQlNXjkFrpcg4
# Seq2Seq_DA (135MB)
gdown https://drive.google.com/uc?id=14y7dXkAEwB_q81cDCow8JNKD2aAAPxbW
```


## Experimental

A trainer based on the [Decision Transformer](https://arxiv.org/abs/2106.01345) has been added.

The results are underwhelming (below the best Seq2Seq) but it constitutes a good starting point for
anybody wanting to test Transformer in VLN-CE.

Pretrained models with corresponding training file under :


[Decision Transformer Agent](https://drive.google.com/file/d/1-E1l5g7DM36m3HYx8b4b4CNBC8d-OS83/view?usp=sharing)

[Enhanced Decision Transformer Agent](https://drive.google.com/file/d/1b2hpkHpiZIc2CBsaLzZWCa7qurfKDDsu/view?usp=sharing)

[Full Decision Transformer Agent](https://drive.google.com/file/d/1rS2_yo9_z35zzaHW4CtByorZ-jpDpht_/view?usp=sharing)

