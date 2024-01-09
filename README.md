# AlignTDS
Analyzing LLM Alignment via Token distribution shift. 

This is part of the Re-Align project by AI2 Mosaic. Please find more information on our website: [https://allenai.github.io/re-align/](https://allenai.github.io/re-align/index.html) and our [paper](https://arxiv.org/abs/2312.01552).

 

## Alignment as Token Distribution Shifts

  
[![Alignment Image](https://allenai.github.io/re-align/images/urial_tds_short.png)](https://allenai.github.io/re-align/images/urial_tds_short.png)


### How can we know what are changed by alignment tuning?

Our analysis is based on token distribution shifts (TDS).

#### The pipeline is as follows:

1. We choose a pair of base and aligned LLMs (e.g., Llama-2 and Llama-2-chat).
2. Given a user query (i.e., instruction) **q**, we first input it to the aligned LLM and get its answer (via greedy decoding). We call this answer from the aligned model as **o**={o<sub>1</sub>, o<sub>2</sub>, ..., o<sub>T</sub>}. We save the distribution at each position **t**, which is named **P<sub>aligned</sub>**.
3. For each token position **t**, we use the context {**q**, o<sub>1</sub>, o<sub>2</sub>, ..., o<sub>t-1</sub>} as input to the base LLM (untuned) and get the token distribution of the base LLM for the next position **t**. Let's name this distribution **P<sub>base</sub>**.
4. Now, we can analyze what are changed by alignment tuning through the difference between **P<sub>aligned</sub>** and **P<sub>base</sub>** at each position!

#### To easily analyze the TDS at each position, we define three types of positions based on the rank of aligned token (**o<sub>t</sub>**) in the token list ranked by **P<sub>base</sub>**:

1. **Unshifted positions:** the aligned token (i.e., top 1 from **P<sub>aligned</sub>**) is also the top 1 token from **P<sub>base</sub>**
2. **Marginal positions:** the aligned token is within the 2nd or 3rd tokens ranked by **P<sub>base</sub>**.
3. **Shifted positions:** the aligned token's rank is not within the top 3 tokens from **P<sub>base</sub>**.

---

#### Web Demos for TDS analysis

**You can visualize the token distribution shifts easily with our web demos:**

- TDS demo: [Llama-2-7b vs Llama-2-7b-chat](tds/llama2/) (shifted ratio: **7.8%**)
- TDS demo: [Llama-2-7b vs Vicuna-7b-v1.5](tds/vicuna/) (shifted ratio: **4.8%**)
- TDS demo: [Mistral-7b vs Mistral-7b-instruct](tds/mistral/) (shifted ratio: **5.2%**)



---

### Key Findings:

1. **Alignment affects only a very small fraction of tokens.** The base and aligned LLMs behave the same in decoding on most positions, where they share the same top-ranked tokens.
2. **Alignment mainly concerns stylistic tokens,** such as discourse markers, transitional words, and safety disclaimers, which only take about 5-8% of the positions.
3. **Alignment is more critical for earlier tokens.** For most positions, the aligned model's top-ranked token is within the top 5 tokens ranked by the base model.
4. **Base LLMs have already acquired adequate knowledge to follow instructions.** They behave very similarly to aligned LLMs when given an appropriate context as a prefix.

---

### Token Distribution Shift Analysis

#### 1️⃣ Knowledge-intensive content originates from base LLMs.

<details>
  <summary>Click to show/hide image</summary>
  
  ![Image 1](https://allenai.github.io/re-align/images/tds_1.png)
</details>

#### 2️⃣ Token distribution shifts on different pairs of LLMs.

<details>
  <summary>Click to show/hide images</summary>

  ![Image 2](https://allenai.github.io/re-align/images/figure8.png)
  ![Image 2](https://allenai.github.io/re-align/images/tds_2.png)
</details>

#### 3️⃣ What does alignment tuning learn?

<details>
  <summary>Click to show/hide image</summary>

  ![Image 3](https://allenai.github.io/re-align/images/tds_3.png)
</details>

#### 4️⃣ Token distribution shift diminishes over time during decoding.

<details>
  <summary>Click to show/hide images</summary>

  ![Image 4](https://allenai.github.io/re-align/images/urial_tds_curve.png)
  ![Image 4](https://allenai.github.io/re-align/images/tds_4.png)
</details>



## Steps 

### Generate outputs from aligned models 

Here we use a generated output file that contains the outputs of Llama-2-7b-chat on just-eval-instruct. 
Filepath of an example data: `data/Llama-2-7b-chat-hf.json`
Please see how to generate outputs from aligned models in https://github.com/re-align/URIAL/ .

### Run Logit analysis 

#### Save the token Logits of aligned models 
```bash 
# i2i   
instruct_data_file="data/Llama-2-7b-chat-hf.json"
logits_folder="saved_logits/just_eval_1000/llama2/shards/"
# i2i
mkdir -p $logits_folder 
n_shards=4 # or 1 if you only have one gpu
shard_size=250 # or 1000 if you only have one gpu
start_gpu=0
for ((start = 0, end = (($shard_size)), gpu = $start_gpu; gpu < $n_shards+$start_gpu; start += $shard_size, end += $shard_size, gpu++)); do
    CUDA_VISIBLE_DEVICES=$gpu python src/logit_analysis.py \
                --data_file $instruct_data_file \
                --logits_folder $logits_folder \
                --pair llama \
                --mode i2i \
                --start $start --end $end &  
done
# Merge the shards
python src/scripts/merge_logits.py saved_logits/just_eval_1000/llama/ llama i2i
```


#### Save the token logits of base models
```bash 
logits_folder="saved_logits/just_eval_1000/llama2_tp/shards/"
mkdir -p $logits_folder
n_shards=4
shard_size=250
start_gpu=0
for ((start = 0, end = (($shard_size)), gpu = $start_gpu; gpu < $n_shards+$start_gpu; start += $shard_size, end += $shard_size, gpu++)); do
    CUDA_VISIBLE_DEVICES=$gpu python src/logit_analysis.py \
                --data_file $instruct_data_file \
                --enable_template \
                --logits_folder $logits_folder \
                --pair llama2 \
                --mode i2b \
                --i2i_pkl_file saved_logits/just_eval_1000/llama2/llama2-i2i.pkl \
                --start $start --end $end & 
done
# Merge the shards
python src/scripts/merge_logits.py saved_logits/just_eval_1000/llama2_tp/ llama2 i2b
```


#### Data Reformatting
```bash 
python src/demo/data_prep.py llama2_tp saved_logits/just_eval_1000/llama2/llama2-i2i.pkl saved_logits/just_eval_1000/llama2_tp/llama2-i2b.pkl
```


#### Generate HTML pages for visualization
```bash
python src/demo/generate_html.py llama2_tp
```


## Citation 

```bibtex
@article{Lin2023ReAlign,
    author = {Bill Yuchen Lin and Abhilasha Ravichander and Ximing Lu and Nouha Dziri and Melanie Sclar and Khyathi Chandu and Chandra Bhagavatula and Yejin Choi},
    journal = {ArXiv preprint},
    title = {The Unlocking Spell on Base LLMs: Rethinking Alignment via In-Context Learning},
    year = {2023}
}
```
