import json 
from models import DecoderOnlyModelManager
import torch
import torch.nn.functional as F
import argparse
import os 
from tqdm import tqdm 
import pickle 


model_pairs = { 
    "vicuna": {"i2b": "meta-llama/Llama-2-7b-hf", "i2i":"lmsys/vicuna-7b-v1.5"},
    "llama2": {"i2b": "meta-llama/Llama-2-7b-hf", "i2i":"meta-llama/Llama-2-7b-chat-hf"},
    "mistral": {"i2b": "mistralai/Mistral-7B-v0.1", "i2i":"mistralai/Mistral-7B-Instruct-v0.1"},
    # "llama2-13b": {"base": "meta-llama/Llama-2-13b-hf", "instruct":"meta-llama/Llama-2-13b-chat-hf"},
}



def parse_args():
    parser = argparse.ArgumentParser()  
    parser.add_argument('--pair', default="llama2", type=str)
    parser.add_argument('--mode', default="i2i", type=str) # i2i or i2b
    # parser.add_argument('--probe', default="self", type=str)
    parser.add_argument('--top_k', default=30, type=int)
    parser.add_argument('--start', default=1, type=int)
    parser.add_argument('--end', default=5, type=int)
    # parser.add_argument('--base_data_file', default="result_dirs/just_ours/K=0+N=fixed+sample=3/Llama-2-7b-hf.json", type=str)
    parser.add_argument('--data_file', default="result_dirs/just_results/Llama-2-7b-chat-hf.json", type=str)
    parser.add_argument('--i2i_pkl_file', type=str)
    parser.add_argument('--logits_folder', default="saved_logits/just_eval/shards/", type=str)
    parser.add_argument('--enable_template', action="store_true")
    return parser.parse_args()


def get_logits(prompt, generated_text, model, tokenizer, device, top_k, mode="i2i"): 
    # decoded_tokens = tokenizer.decode(output[0], skip_special_tokens=False)
    generated_tokens = tokenizer.encode(generated_text, return_tensors="pt", add_special_tokens=False).to(device)
    # print(generated_tokens)
    # print([tokenizer.decode(generated_tokens[0])])
    results = []
    input_ids = tokenizer.encode(prompt, return_tensors='pt', add_special_tokens=True).to(device)
    final_generated_tokens = []
    for i in range(len(generated_tokens[0])):
        with torch.no_grad():
            model_outputs = model(input_ids)
            logits = model_outputs.logits
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            # logits = top_k_top_p_filtering(logits, top_k=top_k)
            # print(logits)
            top_k_indices = logits.topk(k=top_k).indices
        # print(top_k_indices)
        result = []
        logits = logits.cpu().numpy()
        probs = probs.cpu().numpy()
        top_k_indices = top_k_indices.cpu().numpy()
        # print(input_ids)
        prefix = tokenizer.decode(input_ids[0]) # TODO: check results
        for j in range(top_k):
            d = {}
            d["token_id"] = top_k_indices[0][j]
            d["token"] = tokenizer.decode(d["token_id"])
            d["logit"] = logits[0][d["token_id"]]
            d["prob"] = probs[0][d["token_id"]]
            result.append(d)
        results.append({"tokens":result, "prefix": prefix})
        # print("---->", results[-1][:5])
        selected_token = generated_tokens[0][i]
        if mode == "i2i":
            # assert selected_token == top_k_indices[0][0] 
            # selected_token = top_k_indices[0][0]  # TODO: this can cause instability  
            final_generated_tokens.append(selected_token)
        new_token_tensor = torch.tensor([[selected_token]]).to(device)
        # print(tokenizer.decode(generated_tokens[0][i]))
        input_ids = torch.cat((input_ids, new_token_tensor), dim=1).to(device)
        # print([tokenizer.decode(input_ids[0])], end=" ")
    # for position, tokens in enumerate(results):
    #     print(tokens)
    if mode == "i2i":
        final_generated_text = tokenizer.decode(final_generated_tokens, add_special_token=False)
        # print(generated_text, final_generated_text)
        return results, final_generated_text
    elif mode == "i2b":
        return results
    else:
        raise NotImplementedError
    


def main():

    args = parse_args()

    cache_dir = None 
    
    model_path = model_pairs[args.pair][args.mode]
    model_name = "x"
      
    with open(args.data_file) as f:
        instruct_data = json.load(f) 
        
    ids = []
    input_texts = [] 
    output_texts = [] 
    for ind in range(len(instruct_data)): 
        item = instruct_data[ind] 
        ids.append(item["id"])
        if args.mode == "i2i":
            input_texts.append(item["input"]) 
        elif args.mode == "i2b":
            # TODO: add the template here
            in_text = item["pure_input"]
            if args.enable_template:
                in_text = f"# Query: \n ```{in_text}``` \n\n \n# Answer: \n ```"
            input_texts.append(in_text) 
            
        else:
            raise NotImplementedError
        
        output_texts.append(item["output"][0])
        
    print(f"model_path={model_path}") 
    mm = DecoderOnlyModelManager(model_path, model_name, cache_dir)
    mm.load_model()
    model = mm.model
    device = model.device
    tokenizer = mm.tokenizer 
    logit_results = [] 
     
    if args.mode == "i2b":
        output_texts = [] 
        assert os.path.exists(args.i2i_pkl_file)
        with open(args.i2i_pkl_file, "rb") as file:
            i2i_results = pickle.load(file) 
        for d in i2i_results:
            output_texts.append(d["final_generated_text"])
     
    
    s = args.start
    e = args.end
    # print(f"ID: {ids[s]} --> Example Input: {input_texts[s]}; Example Output: {output_texts[s]}")
    for ind, prompt, generated_text in tqdm(zip(ids[s:e], input_texts[s:e], output_texts[s:e]), total=e-s, desc=args.mode): 
        d = {"id": ind, "prompt": prompt, "probe_text": generated_text}
        
        r = get_logits(prompt, generated_text, model, tokenizer, device, args.top_k, mode=args.mode)
        if args.mode == "i2i":
            d["results"], d["final_generated_text"] = r
        elif args.mode == "i2b":
            d["results"] = r
        logit_results.append(d) 
        # print()
    # Save logit_results to i2i.pkl
    with open(os.path.join(args.logits_folder, f"{args.pair}-{args.mode}.[{args.start}:{args.end}].pkl"), "wb") as file:
        pickle.dump(logit_results, file) 
    
if __name__ == "__main__":
    main()