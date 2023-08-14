import argparse
import os
import torch
import numpy as np
from load import load_model
from transformers import AutoModelForCausalLM, LlamaForCausalLM, AutoTokenizer
from transformer_lens import HookedTransformer, utils
from peft import PeftModel


def compute_and_save_weight_statistics(args):
    
    model_name = "meta-llama/Llama-2-7b-hf" ## Model_name - name in huggingface
    adapters_name = "/nfs/aip/llm/PEFT_llama2/checkpoint-29892"  ## adapters_name - path to checkpoint
    model = AutoModelForCausalLM.from_pretrained(model_name)## 
    #m = PeftModel.from_pretrained(m, adapters_name) ## adapters_name - path to checkpoint

    #model = m.merge_and_unload()

    tokenizer = AutoTokenizer.from_pretrained('meta-llama/Llama-2-7b-hf')

    model = HookedTransformer.from_pretrained("meta-llama/Llama-2-7b-hf", hf_model=model, device="cpu", fold_ln=False, center_writing_weights=False, center_unembed=False, tokenizer=tokenizer)
    model.eval()

    in_norm = model.W_in.norm(dim=1).detach().numpy()
    in_bias = model.b_in.detach().numpy()
    out_norm = model.W_out.norm(dim=-1).detach().numpy()
    out_bias = model.b_out.detach().numpy()
    cos = torch.nn.CosineSimilarity()(model.W_in.detach(), torch.swapaxes(model.W_out.detach(), 1, 2))

    n_layers, n_neurons = in_norm.shape
    statistics = np.zeros((5, n_layers, n_neurons))
    statistics[0] = in_norm
    statistics[1] = in_bias
    statistics[2] = out_norm
    statistics[3, :, :len(out_bias[0])] = out_bias
    statistics[4] = cos

    save_dir = '/home/ewolos/sparse-probing-paper/saved_stats/'
    os.makedirs(save_dir, exist_ok=True)
    save_file = os.path.join(save_dir, f'llama2-vanilla-stats.npy')
    np.save(save_file, statistics)


def load_weight_statistics(model_name, save_dir=None):
    if save_dir is None:
        save_dir = os.path.join(
            os.environ.get('RESULTS_DIR', 'results'),
            'weight_statistics'
        )
    stats = np.load(os.path.join(save_dir, f'{model_name}.npy'))
    _, _, n_neurons = stats.shape
    return {
        'in_norm': stats[0],
        'in_bias': stats[1],
        'out_norm': stats[2],
        'out_bias': stats[3, :, :n_neurons//4],
        'cos': stats[4]
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    #parser.add_argument('--model_name', type=str, help='Model name')
    args = parser.parse_args()
    compute_and_save_weight_statistics(args)
