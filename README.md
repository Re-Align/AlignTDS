# AlignTDS 
Analyzing the Alignment of LLMs through the lens of Token Distribution Shift (TDS). Part of the Re-Align project by AI2 Mosaic. More info on our [website](https://allenai.github.io/re-align/index.html) and in our [paper](https://arxiv.org/abs/2312.01552).

## Alignment as Token Distribution Shifts 🔄

![Alignment Image](https://allenai.github.io/re-align/images/urial_tds_short.png)

### What changes does alignment tuning bring? 🧐

> **Analysis of TDS**: Our approach involves comparing token distributions between base and aligned Large Language Models (LLMs) to understand the impact of alignment tuning.

#### The Analysis Pipeline ⚙️

1. Choose a pair of LLMs (e.g., Llama-2 and Llama-2-chat).
2. Get answer **o** from the aligned LLM.
3. Input the context to the base LLM and get the token distribution for the next position **P<sub>base</sub>**.
4. Analyze the differences in distribution to understand the effects of alignment tuning.

#### Types of Token Positions Based on TDS 📊

- **Unshifted positions:** 🏠 Aligned token is also top 1 in **P<sub>base</sub>**.
- **Marginal positions:** 🌿 Aligned token ranks 2nd or 3rd by **P<sub>base</sub>**.
- **Shifted positions:** 🚀 Aligned token is outside the top 3 in **P<sub>base</sub>**.

---

#### Web Demos for TDS analysis 🌐

- Visualize token distribution shifts easily with our web demos:
  - TDS demo: [Llama-2-7b vs Llama-2-7b-chat](tds/llama2/) (shifted ratio: **7.8%**)
  - TDS demo: [Llama-2-7b vs Vicuna-7b-v1.5](tds/vicuna/) (shifted ratio: **4.8%**)
  - TDS demo: [Mistral-7b vs Mistral-7b-instruct](tds/mistral/) (shifted ratio: **5.2%**)

### Key Findings 🔑

1. **Only a small fraction of tokens are affected by alignment.** The base and aligned LLMs usually share the same top-ranked tokens.
2. **Alignment mainly changes stylistic elements,** around 5-8% of positions.
3. **Earlier tokens are more critical for alignment.** The top token of the aligned model is often in the top 5 of the base model.
4. **Base LLMs are already primed to follow instructions** given an appropriate context.

### Token Distribution Shift Analysis 

1. **Knowledge content comes from base LLMs.**

<details>
  <summary>Click to show/hide image 🖼️</summary>
  
  ![Knowledge Content Image](https://allenai.github.io/re-align/images/tds_1.png)
</details>

2. **TDS across different LLM pairs.**

<details>
  <summary>Click to show/hide images 🖼️</summary>

  ![TDS Comparison Image](https://allenai.github.io/re-align/images/figure8.png)
  ![TDS Pair Image](https://allenai.github.io/re-align/images/tds_2.png)
</details>

3. **Learnings from alignment tuning.**

<details>
  <summary>Click to show/hide image 🖼️</summary>

  ![Alignment Learning Image](https://allenai.github.io/re-align/images/tds_3.png)
</details>

4. **TDS diminishes over time during decoding.**

<details>
  <summary>Click to show/hide images 🖼️</summary>

  ![TDS Diminishing Image](https://allenai.github.io/re-align/images/urial_tds_curve.png)
  ![TDS Over Time Image](https://allenai.github.io/re-align/images/tds_4.png)
</details>

## Procedures 🛠️

### Generate outputs from aligned models 

We use a generated output file containing the responses of aligned models. Filepath example: `data/Llama-2-7b-chat-hf.json`.
See the repo [URIAL](https://github.com/re-align/URIAL) for generation details.

### Run Logit Analysis 📊

#### Save the token logits from models

```bash
# Sample bash code for saving logits
```

#### Data Reformatting and Visualization 👁️

```bash
# Commands for data reformatting and generating HTML for visualization
```

### TODOs 📝

- [ ] Integrate model generation into the logit computation process.
- [ ] Use vllm lib for efficiency improvements.
- [ ] Create an interactive demo.
- [ ] Add more data from larger LLMs.
- [ ] Compare models fine-tuned in different ways.

## Citation 📄

```bibtex
@article{Lin2023ReAlign,
    author = {Bill Yuchen Lin and Abhilasha Ravichander and Ximing Lu and Nouha Dziri and Melanie Sclar and Khyathi Chandu and Chandra Bhagavatula and Yejin Choi},
    journal = {ArXiv preprint},
    title = {The Unlocking Spell on Base LLMs: Rethinking Alignment via In-Context Learning},
    year = {2023}
}
```
