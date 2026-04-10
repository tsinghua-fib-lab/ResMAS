# LLM_MAS_resilience

The official PyTorch implementation of "[ResMAS: Resilience Optimization in LLM-based Multi-agent Systems](https://ojs.aaai.org/index.php/AAAI/article/view/40824/)" (AAAI'26).

## Usage

### Stage 0: Preparation
- Download Qwen2.5-7B-Instruct from https://www.modelscope.cn/models/qwen/Qwen2.5-7B-Instruct, and put it into the "Qwen2.5-7B-Instruct" folder.
- Install LLaMA-Factory from https://github.com/hiyouga/LLaMA-Factory
- Set API keys in llm_configs.py
- Use the ``pip install -r requirements.txt`` command to install packages used in this project

### Stage 1: Topology Optimization

**1. Supervised fine-tuning**
```
cd sft
bash run_sft.sh
```

**2. Training reward model**
```
cd reward_model
bash run.sh
```

**3. GRPO training**
```
cd GRPO
bash run.sh
```

**4. Generate MAS topology**
```
cd GRPO
bash run_api.sh
python generate_topology.py
```


### Stage 2: Prompt Optimization

```
cd prompt_optimization
MATH: python run_math_optim.py
MMLU: python run_mmlu_optim.py
Chess: python run_chess_optim.py
```

### Stage 3: Evaluation
```
cd evaluation
bash run.sh
```

## Note

If you found this library useful in your research, please consider citing:

```
@article{zhou2026resmas, 
    title={ResMAS: Resilience Optimization in LLM-based Multi-agent Systems}, 
    volume={40}, 
    url={https://ojs.aaai.org/index.php/AAAI/article/view/40824}, 
    DOI={10.1609/aaai.v40i41.40824}, 
    number={41}, 
    journal={Proceedings of the AAAI Conference on Artificial Intelligence}, 
    author={Zhou, Zhilun and Liu, Zihan and Liu, Jiahe and Shao, Qingyu and Wang, Yihan and Shao, Kun and Jin, Depeng and Xu, Fengli}, 
    year={2026}, 
    month={Mar.}, 
    pages={35176-35184} 
}
```

