import pickle 
import json 

import sys
import os
from tqdm import tqdm 



# Get the current file's directory
current_dir = os.path.dirname(os.path.abspath(__file__))

# Get the parent directory by going up one level
parent_dir = os.path.dirname(current_dir)

# Add the parent directory to sys.path
sys.path.append(parent_dir)

# Now you can import the module from the parent folder
from compute_dist_diff import compute_similarity


pair_name = sys.argv[1]
# pair_name = "vicuna_tp"

os.system(f"mkdir docs/{pair_name}_justeval/")

with open(f"src/demo/just_eval+{pair_name}.pkl", "rb") as f:
    print("Loading:", f.name)
    all_data = pickle.load(f)


template_file =  "src/demo/template.html"
index_file =  "src/demo/index.html"

with open(template_file) as f:
    template = f.read()
 


token_freqs = {}
token_counts = {"unshifted":0, "shifted": 0, "marginal": 0}
shifted_tokens = {}
unshifted_tokens = {}
diff_points = []

shifted_relative_positions = []

def genertate_page(ind=0, template=None): 
    item = all_data[ind] 

    item["prompt"] = item["prompt"].replace("# Query:", "").replace("# Answer:", "").replace("```", "")
    template = template.replace("#{{prompt}}", item["prompt"])
    template = template.replace("#{{probe_text}}", item["probe_text"])
    # template = template.replace("#{{base_output}}", item["base_output"])

    template = template.replace("#{{instance_id}}", str(ind))

    base_actions = []
    inst_actions = []
    len_tokens = len(item["formatted_base"])
    assert len_tokens == len(item["formatted_inst"])
    position_tables = {}
    for i in range(len_tokens):
        list1 = item["formatted_base"][f"position_{i}"]
        list2 = item["formatted_inst"][f"position_{i}"]
        jaccard = compute_similarity(list1["candidates"], list2["candidates"], metric="jaccard", top_k=10)
        KL = compute_similarity(list1["candidates"], list2["candidates"], metric="KL", top_k=10)
        TR = compute_similarity(list1["candidates"], list2["candidates"], metric="top_rank", top_k=10)
        TP = compute_similarity(list1["candidates"], list2["candidates"], metric="top_prob", top_k=50)
        BTP = compute_similarity(list1["candidates"], list2["candidates"], metric="bi_top_prob", top_k=10)
        # base_action = f'\t\t\t<a href="#" class="action" data-target="position_{i}">{list1["selected_token"]}<sub style="font-size:7pt">({i})</sub></a>'
        # base_actions.append(base_action)

        style = " "
        # if list1["selected_token"] != list2["selected_token"]:
        #     style = "font-weight: 800; color:red"
        # else:
        # style = f"font-weight: {300+500*(1-jaccard)}; color: rgba(255, 0, 0, {1-jaccard+0.2});"
        
        if TR < 1:
            style = f"font-weight: {400}; color: blue;"
            token_counts["unshifted"] += 1
            if list2["selected_token"] not in unshifted_tokens:
                unshifted_tokens[list2["selected_token"]] = 0
            unshifted_tokens[list2["selected_token"]] += 1
        elif TR < 3:
            style = f"font-weight: {400}; color: rgba(155, 100, 0, {1});"
            token_counts["marginal"] += 1
        else:
            style = f"font-weight: {400}; color: rgba(255, 0, 0, {1});"
            token_counts["shifted"] += 1
            if list2["selected_token"] not in shifted_tokens:
                shifted_tokens[list2["selected_token"]] = 0
            shifted_tokens[list2["selected_token"]] += 1
            # save the relative position of the shifted token in the whole sequence 
            shifted_relative_positions.append(i/len_tokens)
        
        # increase freq 1 for the token
        token_freqs[list2["selected_token"]] = 1 + token_freqs.get(list2["selected_token"], 0)
        diff_points.append((i/len_tokens, jaccard))
            
        if list1["selected_token"] != list2["selected_token"]:
            style += "text-decoration: underline;"
        
        inst_action = f'\t\t\t<a href="#" class="action" data-target="position_{i}" style="{style}">{list2["selected_token"]}</a>' # <sub>({i})</sub>
        inst_actions.append(inst_action) 
        table_str = ""
        # The content for Prefix  

        table_str += f'<table><tr><th style="background-color:orange">Base@{i}</th>  <th>Aligned@{i}</th></tr>'
        # table_str += "<tr><td>BaseLM</td><td>InstLM</td></tr>"
        count = 0
        for bas, ins in zip(list1["candidates"], list2["candidates"]):
            if bas['token'] == list2["candidates"][0]["token"]:
                table_str += f"<tr><td><u>{bas['token']}</u> ({bas['norm_prob']*100 :.2f}%)</td>"
            else:
                table_str += f"<tr><td>{bas['token']} ({bas['norm_prob']*100 :.2f}%)</td>"
            if count == 0:
                table_str += f"<td><b>{ins['token']}</b> ({ins['norm_prob']*100 :.2f}%)</td></tr>"
            else:
                table_str += f"<td>{ins['token']} ({ins['norm_prob']*100 :.2f}%)</td></tr>"
            count += 1 
            if count >= 10:
                break 
        table_str += "</table>"
        table_str += "<div class='prefix'>"
        # table_str += f'<b>TP@30</b>: {TP:.4f} &nbsp; | &nbsp;'
        # table_str += f'<b>TR@10</b>: {TR:.4f} &nbsp; | &nbsp;'
        table_str += f'<b>KL@10</b>: {KL:.4f} &nbsp; | &nbsp;'
        table_str += f'<b>Jaccard@10</b>: {jaccard:.4f} &nbsp; | &nbsp;'
        table_str += f'<b>WS@10</b>: {compute_similarity(list1["candidates"], list2["candidates"], metric="weighted_sum", top_k=10):.4f} &nbsp; | &nbsp;'        
        table_str += "</div> <hr>"
        
        
        table_str += "<div class='prefix' style='background-color:orange;text-align:left;'><span> <b>Base Prefix: </b>" 
        # table_str +=  f'<span style="color:gray">{item["prompt"]}</span>'
        # table_str +=  " ".join([item["formatted_inst"][f"position_{j}"]["selected_token"] for j in range(0, i)])         
        table_str += item["formatted_base"][f"position_{i}"]["prefix"]
        table_str +=  "</span></div> <hr>"
        
        table_str += "<div class='prefix' style='background-color:lightblue;text-align:left;'><span> <b>Aligned Prefix: </b>" 
        # table_str +=  f'<span style="color:gray">{item["prompt"]}</span>'
        # table_str +=  " ".join([item["formatted_inst"][f"position_{j}"]["selected_token"] for j in range(0, i)])         
        table_str += item["formatted_inst"][f"position_{i}"]["prefix"]
        table_str +=  "</span></div> <hr>"
        position_tables[f"position_{i}"] = table_str


    dict_data = json.dumps(position_tables)
    template = template.replace("#{{words_data}}", dict_data)
    template = template.replace("#{{inst_actions}}", "\n".join(inst_actions))
    # template = template.replace("#{{base_actions}}", "\n".join(base_actions))
    #{{next_link}}
    template = template.replace("#{{prev_link}}", f"{ind-1}.html")
    template = template.replace("#{{next_link}}", f"{ind+1}.html")


    with open(f"docs/{pair_name}_justeval/{ind}.html", "w") as f:
        f.write(template.replace("<s><s>", "<s>").replace("<s>", "&lt;s&gt;").replace("</s>", "&lt;/s&gt;").replace("<<", "&lt;&lt;").replace(">>", "&gt;&gt;"))
    return str(ind), item["prompt"]

with open(index_file) as f:
    index_template = f.read()

rows = []
for ind in tqdm(range(1000), desc="Generating HTMLs"):
    ind, prompt = genertate_page(ind, template)
    rows.append(f'<tr><td>ID: <a href="{ind}.html">{ind}</a></td><td><a href="{ind}.html">{prompt[:100]+" ..."}</a></td></tr>')

index_template = index_template.replace("#{{rows}}", "\n".join(rows))

with open(f"docs/{pair_name}_justeval/index.html", "w") as f:
    f.write(index_template)
    
exit()  
    
print(token_counts)

common_unshifted_tokens = sorted(unshifted_tokens.items(), key=lambda x:x[1], reverse=True)
common_shifted_tokens = sorted(shifted_tokens.items(), key=lambda x:x[1], reverse=True)

shifted_list = []
for t, f in common_shifted_tokens[:50]:
    shifted_list.append(f"'{t}'")
print(", ".join(shifted_list))

# compute the percentage of each token being shifted 
shifted_ratio = {}
for t in token_freqs:
    shifted_ratio[t] = shifted_tokens.get(t, 0)/token_freqs[t]
    
# sort the shifted ratio by the value 
sorted_shifted_ratio = sorted(shifted_ratio.items(), key=lambda x:x[1], reverse=True)

# # save it to a tsv file 
# with open(f"docs/{pair_name}_justeval/shift_ratio_ranked.tsv", "w") as f:
#     for t, r in sorted_shifted_ratio:
#         f.write(f"{t}\t{r}\t{token_freqs[t]}\n")
        
# with open(f"docs/{pair_name}_justeval/diff_points.tsv", "w") as f:
#     for item in diff_points:
#         f.write(f"{item[0]}\t{item[1]}\n")
        


# unshifted_list = []
# for t, f in common_unshifted_tokens[:50]:
#     unshifted_list.append(f"'{t}'")
# print(", ".join(unshifted_list))

# with open(f"docs/{pair_name}_justeval/shift_position_dist.txt", "w") as f:
#     f.write("\n".join([str(x) for x in shifted_relative_positions]))