# ufet_with_domains
Code of paper "Ultra-Fine Entity Typing with Prior Knowledge about Labels: A Simple Clustering Based Strategy", accepted by EMNLP 2023 Findings

# Requirements
- Python 3.7
- transformers == 4.18.0
- skikit-learn == 1.0.2
- faiss == 1.7.2
- pyarrow == 10.0.1

# Run domain models step by step
### Step 1. Run clustering.py to get domains

```
python ./src/domain_model/clustering.py -type_emb [embeddings file name of entity types] -out_file [output domain file name] -clu_type all
```

### Step 2. Run preprocessing.py to build a mapping from domains to original types

```
python ./src/domain_model/preprocessing.py -dom [domain file name got from step 1] -goal open -out_dir [output directory of the mapping]
```

### Step 3. Run find_cn.py to obtain concept neighbor pairs

```
python ./src/domain_model/find_cn.py -goal open -o [output file name of concept neighbor pairs] -d crowd -e [embeddings file name of entity types] -k 5
```

### Step 4. Run get_init_weights.py to get the initial weights of domain types

```
python ./src/domain_model/get_init_weights.py -model [file name got from baseline model] -mapping [mapping file name got from step 2] -out_dir [output directory of initial weights]
```

### Step 5. Add type path to VOCAB_DIR_DICT in constant.py, for example

```
VOCAB_DIR_DICT = {"nb": os.path.join(DATA_ROOT, 'ontology/domain/numberbatch')}
```

### Step 6. Run main.py to run the domain model

```
python ./src/domain_model/main.py -model_id [model name] -d_goal [domain goal, e.g. nb in step 5] -dfn_postfix [postfix of domain type file name] -dfc_param [file name of initial weights for last fc layer] -bert_param [file name of initial weights for BERT] -in_dim [input dimension] -cn
```

