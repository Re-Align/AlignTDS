# AlignTDS
Analyzing LLM Alignment via Token distribution shift. 

This is part of the Re-Align project by AI2 Mosaic. Please find more information on our website: [https://allenai.github.io/re-align/](https://allenai.github.io/re-align/index.html) and our [paper](https://arxiv.org/abs/2312.01552).

 

## Alignment as Token Distribution Shifts

  
[![Alignment Image](https://allenai.github.io/re-align/images/urial_tds_short.png)](https://allenai.github.io/re-align/images/urial_tds_short.png)


**How can we know _what is changed by alignment tuning_ (i.e., instruction tuning via SFT and preference learning via RLHF)?**

...

<details>
  <summary>Click to show/hide details of TDS analysis</summary>

**To easily analyze the TDS at each position, we define three types of positions based on the rank of aligned token (**oₜ**) in the token list ranked by **P<sub>base</sub>**:**

1. **Unshifted positions:** the aligned token (i.e., top 1 from **P<sub>aligned</sub>**) is also the top 1 token from **P<sub>base</sub>**.
2. **Marginal positions:** the aligned token is within the 2nd or 3rd tokens ranked by **P<sub>base</sub>**.
3. **Shifted positions:** the aligned token's rank is not within the top 3 tokens from **P<sub>base</sub>**.
</details>

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




## Generate outputs from aligned models 

Here we use a generated output file that contains the outputs of Llama-2-7b-chat on just-eval-instruct. 
Filepath of an example data: `data/Llama-2-7b-chat-hf.json`
Please see how to generate outputs from aligned models in https://github.com/re-align/URIAL/ .

## Run Logit analysis 

### Save the token Logits of aligned models 
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


### Save the token logits of base models
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


### Data Reformatting
```bash 
python src/demo/data_prep.py llama2_tp saved_logits/just_eval_1000/llama2/llama2-i2i.pkl saved_logits/just_eval_1000/llama2_tp/llama2-i2b.pkl
```


### Generate HTML pages for visualization
```bash
python src/demo/generate_html.py llama2_tp
```
