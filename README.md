# AlignTDS
Analyzing LLM Alignment via Token distribution shift 


## Generate outputs from aligned models 

Here we use a generated output file that contains the outputs of Llama-2-7b-chat on just-eval-instruct. 
Filepath: `data/Llama-2-7b-chat-hf.json`
See how to generate outputs from aligned models in https://github.com/re-align/URIAL/

## Run Logit analysis 

### Save the token logits of aligned models 
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
