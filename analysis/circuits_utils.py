import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import torch
from transformers import pipeline

update_layout_set = {
    "xaxis_range", "yaxis_range", "hovermode", "xaxis_title", "yaxis_title", "colorbar", "colorscale", "coloraxis", "title_x", "bargap", "bargroupgap", "xaxis_tickformat",
    "yaxis_tickformat", "title_y", "legend_title_text", "xaxis_showgrid", "xaxis_gridwidth", "xaxis_gridcolor", "yaxis_showgrid", "yaxis_gridwidth", "yaxis_gridcolor",
    "showlegend", "xaxis_tickmode", "yaxis_tickmode", "xaxis_tickangle", "yaxis_tickangle", "margin", "xaxis_visible", "yaxis_visible", "bargap", "bargroupgap"
}
def imshow(tensor, renderer=None, **kwargs):
    kwargs_post = {k: v for k, v in kwargs.items() if k in update_layout_set}
    kwargs_pre = {k: v for k, v in kwargs.items() if k not in update_layout_set}
    facet_labels = kwargs_pre.pop("facet_labels", None)
    border = kwargs_pre.pop("border", False)
    if "color_continuous_scale" not in kwargs_pre:
        kwargs_pre["color_continuous_scale"] = "RdBu"
    if "margin" in kwargs_post and isinstance(kwargs_post["margin"], int):
        kwargs_post["margin"] = dict.fromkeys(list("tblr"), kwargs_post["margin"])
    fig = px.imshow(tensor.detach().cpu().numpy(), color_continuous_midpoint=0.0, **kwargs_pre)
    if facet_labels:
        for i, label in enumerate(facet_labels):
            fig.layout.annotations[i]['text'] = label
    if border:
        fig.update_xaxes(showline=True, linewidth=1, linecolor='black', mirror=True)
        fig.update_yaxes(showline=True, linewidth=1, linecolor='black', mirror=True)
    # things like `xaxis_tickmode` should be applied to all subplots. This is super janky lol but I'm under time pressure
    for setting in ["tickangle"]:
      if f"xaxis_{setting}" in kwargs_post:
          i = 2
          while f"xaxis{i}" in fig["layout"]:
            kwargs_post[f"xaxis{i}_{setting}"] = kwargs_post[f"xaxis_{setting}"]
            i += 1
    fig.update_layout(**kwargs_post)
    fig.show(renderer=renderer)

def imshow_attn(O, V, Q, K, red_to_blue=False):
    min_val = min(O.min(), V.min(), Q.min(), K.min()).item()
    max_val = max(O.max(), V.max(), Q.max(), K.max()).item()
    
    is_bin = min_val == 0 and max_val == 1
    # Create the figure with subplots
    fig = make_subplots(rows=1, cols=4, subplot_titles=["Q", "K", "V", "O"])
    
    # Prepare the data for each heatmap
    data = [Q, K, V, O]
    for i, matrix in enumerate(data, 1):
        matrix_np = matrix.detach().cpu().numpy()
        colorscale = "Blues" if is_bin and not red_to_blue else "RdBu"
        fig.add_trace(go.Heatmap(z=matrix_np, colorscale=colorscale, zmin=min_val, zmax=max_val, showscale=(i == 4)), row=1, col=i)
    
    # x-axis label
    fig.update_yaxes(title_text="Layer", row=1, col=1)
    for i in range(1, 5):
        fig.update_xaxes(title_text="Head", row=1, col=i)
    
    # Show the plot
    fig.show()
    
def merge_lora(module):
  try:
    A = module.lora_A.default.weight
    B = module.lora_B.default.weight
  except:
    return module.weight
  return (B @ A)
 
    
def lora_norm(module, n_heads, d_head=128, d_model=4096):
  return merge_lora(module).reshape(n_heads, d_head, d_model).norm(dim=(1,2))


def get_lora_attn_norm(layers, normalize=True, n_heads=32, n_kv_heads=8):
  diffK, diffQ, diffV, diffO = [], [], [], []
  for layer in layers:
    diffK.append(lora_norm(layer.self_attn.k_proj, n_kv_heads))
    diffV.append(lora_norm(layer.self_attn.v_proj, n_kv_heads))
    diffQ.append(lora_norm(layer.self_attn.q_proj, n_heads))
    #diffO.append(lora_norm(layer.self_attn.o_proj, n_heads))

  diffK = torch.stack(diffK)
  diffQ = torch.stack(diffQ)
  diffV = torch.stack(diffV)
  #diffO = torch.stack(diffO)

  if normalize:
    max_diff = max(diffQ.max(), diffK.max(), diffV.max()) #, diffO.max())
    min_diff = min(diffQ.min(), diffK.min(), diffV.min()) #, diffO.min())

    diffQ = (diffQ - min_diff) / (max_diff - min_diff)
    diffK = (diffK - min_diff) / (max_diff - min_diff)
    diffV = (diffV - min_diff) / (max_diff - min_diff)
    #diffO = (diffO - min_diff) / (max_diff - min_diff)

  return diffK, diffQ, diffV, torch.zeros_like(diffQ)

def generate(model, tokenizer, df):
    pipe = pipeline("text-generation",
                model=model,
                tokenizer=tokenizer,
                device_map="auto",
                max_new_tokens=5,
                batch_size=20)
    
    texts = [row.text for i, row in df.iterrows()]
    out = pipe(texts)
    return out

def eval(generations, data):
    accuracy = 0
    for i, o in enumerate(generations):
        gen = o[0]['generated_text'][len(data.iloc[i].text):]
        accuracy += 1 if data.iloc[i].answer in gen else 0
    return float(accuracy) / len(data)

def zero_out_lora_layerwise(peft_model, layers, keys=["q_proj", "k_proj", "v_proj", "o_proj"]):
    for i, layer in enumerate(peft_model.base_model.model.model.layers):
        if i in layers:
            for key in keys:
                layer.self_attn.__getattr__(key).lora_A.default.weight = torch.nn.Parameter(torch.zeros_like(layer.self_attn.__getattr__(key).lora_A.default.weight))
    return peft_model
  
def zero_out_lora_head(lora, head, d_head=128):
  lora_B = lora.lora_B.default.weight
  P = torch.eye(lora_B.shape[0], device=lora_B.device)
  P[head*d_head:(head+1)*d_head] = 0
  lora.lora_B.default.weight = torch.nn.Parameter(P @ lora_B.data)
  return lora
  
def zero_out_lora(peft_model, configs):
  for key, layer, head in configs:
    zero_out_lora_head(peft_model.base_model.model.model.layers[layer].self_attn.__getattr__(key), head)
  return peft_model
    
    
def display_model_mix(model):
  # print the model's configuration
  n_heads = model.config.num_attention_heads
  d_model = model.config.hidden_size
  n_kv_heads = model.config.num_key_value_heads
  d_head = d_model // n_heads
  n_layers = model.config.num_hidden_layers
  o_mask = torch.ones(n_layers, n_heads)
  q_mask = torch.ones(n_layers, n_heads)
  k_mask = torch.ones(n_layers, n_kv_heads)
  v_mask = torch.ones(n_layers, n_kv_heads)

  for i, layer in enumerate(model.base_model.model.model.layers):
      Q = merge_lora(layer.self_attn.q_proj).reshape(n_heads, d_head, d_model)
      #O = merge_lora(layer.self_attn.o_proj).reshape(n_heads, d_head, d_model)
      K = merge_lora(layer.self_attn.k_proj).reshape(n_kv_heads, d_head, d_model)
      V = merge_lora(layer.self_attn.v_proj).reshape(n_kv_heads, d_head, d_model)
      q_mask[i, :] = (Q == 0).all(dim = (1,2))
      #o_mask[i, :] = (O == 0).all(dim = (1,2))
      k_mask[i, :] = (K == 0).all(dim = (1,2))
      v_mask[i, :] = (V == 0).all(dim = (1,2))
    
  imshow_attn(o_mask, v_mask, q_mask, k_mask)
  
def mask_to_config(mask, key):
  out = []
  n_layers, n_heads = mask.shape
  for i in range(n_layers):
      for j in range(n_heads):
          if mask[i, j] == 0:
              out.append((key, i, j))
  return out

def mask(m, thres=0.5):
  return (m > thres).float()

def mask_percentile(m, percentile):
  return (m > torch.quantile(m, percentile)).float()