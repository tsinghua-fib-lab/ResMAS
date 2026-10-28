# from GRPO_chess, train on chess, math, mmlu
from cgi import test
from datasets import load_dataset, Dataset
from trl import GRPOConfig, GRPOTrainer
import mlflow
import setproctitle
import os
import json
import torch
from torch_geometric.nn import GCNConv, global_mean_pool
import torch.nn.functional as F
from torch_geometric.data import Data, DataLoader
import re
import json_repair
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model, TaskType
from peft import PeftModel, PeftConfig
from utils import *
import random
random.seed(0)
setproctitle.setproctitle("GRPO")


MODEL_NAME = "../Qwen2.5-7B-Instruct" 
reward_model_path = "../reward_model/output/reward_model_all.pth"
train_data_path = "../data/GRPO_train_data.json" #############
output_dir = "GRPO_output/GRPO_result" #############

mlflow_exp_name = "alldataset_lora"
mlflow_run_name = "GRPO" #############

num_train_epochs = 5
learning_rate = 1e-4
save_steps = 50

####################################### load data #######################################
with open(train_data_path, 'r') as f:
    train_data = json.load(f)
# [prompt1, prompt2]

# Convert prompts to a HuggingFace Dataset with empty completions
data = [{"prompt": p['prompt'], "task": p['task']} for p in train_data]
random.shuffle(data)
dataset = Dataset.from_list(data)

####################################### load model #######################################
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    trust_remote_code=True,
)

peft_model_id = "../sft/saves/sft_lora" 
# peft_config = PeftConfig.from_pretrained(peft_model_id)
model = PeftModel.from_pretrained(model, peft_model_id, is_trainable=True)
# model = model.merge_and_unload()

# === LoRA Configuration ===
# lora_config = LoraConfig(
#     r=16,
#     lora_alpha=32,
#     target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
#     lora_dropout=0.05,
#     bias="none",
#     task_type=TaskType.CAUSAL_LM
# )
# model = get_peft_model(model, lora_config)
print("Model wrapped with LoRA.")
model.print_trainable_parameters()


####################################### load reward model #######################################
# device = torch.device('cpu')
reward_model = GNNRegressor(input_dim=384 + 2, hidden_channels=128)
reward_model.load_state_dict(torch.load(reward_model_path))
reward_model.eval()
print(f"======reward_model loaded from {reward_model_path}=======")

####################################### define reward function #######################################
with open("../data/test_questions.json", "r") as f: # calculate resilience on these prompts
    test_questions = json.load(f)
'''{'question': 'xxx', 'task': 'MATH'}'''

def pred_resilience(model, adj_matrix, test_questions, dataset): # dataset: chess, MATH, MMLU_pro
    question_prompts = [x['question'] for x in test_questions if x['task'] == dataset]
    datas = [{'adj_matrix': adj_matrix, 'question_prompt': question_prompt, 'results': [0.0, 0.0, 0.0, 0.0, 0.0]} for question_prompt in question_prompts]
    try:
        assert len(datas) == 128
    except:
        print(f"len(datas) != 128: {len(datas)}")
        print(f"dataset: {dataset}")
        assert False
    graphs = [create_graph(data) for data in datas]
    loader = DataLoader(graphs, batch_size=256, shuffle=False)
    model.eval()
    with torch.no_grad():
        pred_list = []
        for batch in loader:
            # batch = batch.to(device)
            pred = model(batch.x, batch.edge_index, batch.batch)
            pred_list.extend(pred.cpu().numpy().tolist())

    pred_acc = np.mean(pred_list, axis=0)
    resilience = calculate_resilience(pred_acc)
    acc = pred_acc[0]

    return resilience, acc

def single_reward(prompt, completion, task):
    N, M = extract_constraints(prompt)
    try:
        adj_list = json_repair.loads(completion)
        if isinstance(adj_list, list):
            adj_list = adj_list[0]

        num_nodes = len(adj_list)
        adj_matrix = np.zeros((num_nodes, num_nodes), dtype=int)
        node_ids = sorted([int(k) for k in adj_list.keys()])
        id_to_idx = {node_id: idx for idx, node_id in enumerate(node_ids)}
        for node_str, neighbors in adj_list.items():
            i = id_to_idx[int(node_str)]
            for neighbor in neighbors:
                j = id_to_idx[int(neighbor)]
                if i != j:
                    adj_matrix[i, j] = 1
        edge_num = np.sum(adj_matrix)
    except:
        print("parse adj list failed")
        return -1
    
    # format reward
    if num_nodes != N:
        print(f"node num != N: {num_nodes} != {N}")
        return -1
    elif edge_num > M:
        print(f"edge num > M: {edge_num} > {M}")
        # reward = -1
        reward = (M - edge_num) / M
        return reward
    
    ###############################################
    resilience, acc = pred_resilience(reward_model, adj_matrix, test_questions, dataset=task)
    # reward = (resilience + acc) / 2 #######################
    reward = edge_num / M + resilience
    # reward = edge_num / M
    # reward = 1
    return reward

def chess_reward_func(prompts, completions, task, **kwargs):
    rewards = []
    for prompt, completion, t in zip(prompts, completions, task):
        if t == "chess":
            reward = single_reward(prompt, completion, t)
            rewards.append(reward)
        else:
            # Return None for other tasks
            rewards.append(None)
    return rewards

def math_reward_func(prompts, completions, task, **kwargs):
    rewards = []
    for prompt, completion, t in zip(prompts, completions, task):
        if t == "MATH":
            reward = single_reward(prompt, completion, t)
            rewards.append(reward)
        else:
            # Return None for other tasks
            rewards.append(None)
    return rewards

def mmlu_reward_func(prompts, completions, task, **kwargs):
    rewards = []
    for prompt, completion, t in zip(prompts, completions, task):
        if t == "MMLU_pro":
            reward = single_reward(prompt, completion, t)
            rewards.append(reward)
        else:
            # Return None for other tasks
            rewards.append(None)
    return rewards



training_args = GRPOConfig(
    output_dir=output_dir,
    logging_steps=1,
    num_train_epochs=num_train_epochs, # default=3
    num_generations=4,
    per_device_train_batch_size=4,
    max_completion_length=512,
    learning_rate=learning_rate,
    bf16=True,
    save_only_model=True,
    save_steps=save_steps,
    report_to="mlflow",
)


def is_main_process(): 
    # avoid creating multiple mlflow runs when using multiple GPUs
    return os.environ.get("RANK", "0") == "0"

if is_main_process():
    # mlflow.set_tracking_uri('./mlflow_output/') # default is ./mlruns/
    mlflow.set_experiment(mlflow_exp_name)
    mlflow.start_run(run_name=mlflow_run_name)

trainer = GRPOTrainer(
    model=model,
    reward_funcs=[chess_reward_func, math_reward_func, mmlu_reward_func],
    args=training_args,
    train_dataset=dataset,
)
trainer.train()

if is_main_process():
    mlflow.end_run()