# M3F-KG

The code and dataset of paper _**Multi-Modal Knowledge Graph Fusion via Multi-Feature Interaction: A Joint Framework for Entity Alignment and Completion**_. The implementation of the feature interaction part please refer to [MFIEA](https://doi.org/10.1007/s10844-025-00924-w).

## Dataset

### Entity alignment datasets

The multi-modal version of DBP15K dataset comes from the [EVA](https://github.com/cambridgeltl/eva) repository.

We modify the DBP15K dataset to validate the performance of M3F-KG in entity alignment and cross-graph relation prediction tasks. Specifically, We construct cross-graph triples using seed entities to connect the graphs into a unified graph and split the original triples and the cross-graph triples into training and test sets at a ratio of 70% and 30%, respectively.

### Relation prediction datasets

The Relation prediction  datasets (WN18/WN18RR/FB15K-237) comes from [KG-BERT](https://github.com/yao8839836/kg-bert).

## Training M3F-KG

### Entity alignment datasets

Here is the example of training M3F-KG on `DBP15K`.

```bash
bash run_dbp15k.sh 0 zh_en 2026
bash run_dbp15k.sh 0 ja_en 2026
bash run_dbp15k.sh 0 fr_en 2026
```

### Relation prediction datasets

Here is the example of training M3F-KG on `WN18`  `WN18RR`  `FB15K-237`. 

```bash
bash run_wn18.sh 0 2026
bash run_wn18rr.sh 0 2026
bash run_fb15k-237.sh 0 2026
```
> If you have any difficulty or question in running code and reproducing experiment results, please email to lvmenglong@stu.hebust.edu.cn and why_mail4work@163.com.


## Acknowledgement

Our codes are modified based on [EVA](https://github.com/cambridgeltl/eva), [BootEA](https://github.com/nju-websoft/BootEA), [MCLEA](https://github.com/lzxlin/MCLEA), [KG-BERT](https://github.com/yao8839836/kg-bert), and we would like to thank their open-sourced work.

