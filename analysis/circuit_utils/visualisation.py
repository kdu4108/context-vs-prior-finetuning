import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import torch
import numpy as np
import einops

update_layout_set = {
    "xaxis_range", "yaxis_range", "hovermode", "xaxis_title", "yaxis_title", "colorbar", "colorscale", "coloraxis", "title_x", "bargap", "bargroupgap", "xaxis_tickformat",
    "yaxis_tickformat", "title_y", "legend_title_text", "xaxis_showgrid", "xaxis_gridwidth", "xaxis_gridcolor", "yaxis_showgrid", "yaxis_gridwidth", "yaxis_gridcolor",
    "showlegend", "xaxis_tickmode", "yaxis_tickmode", "xaxis_tickangle", "yaxis_tickangle", "margin", "xaxis_visible", "yaxis_visible", "bargap", "bargroupgap"
}
def imshow(tensor, subpart_border=None, invert_y=False, aspect_ratio='auto', **kwargs):
    kwargs_post = {k: v for k, v in kwargs.items() if k in update_layout_set}
    kwargs_pre = {k: v for k, v in kwargs.items() if k not in update_layout_set}
    facet_labels = kwargs_pre.pop("facet_labels", None)
    border = kwargs_pre.pop("border", False)
    if "color_continuous_scale" not in kwargs_pre:
        kwargs_pre["color_continuous_scale"] = "RdBu"
    if "margin" in kwargs_post and isinstance(kwargs_post["margin"], int):
        kwargs_post["margin"] = dict.fromkeys(list("tblr"), kwargs_post["margin"])
    
    
    # Create the figure with specific aspect mode
    fig = px.imshow(tensor.detach().cpu().numpy(), color_continuous_midpoint=0.0, **kwargs_pre)
    fig.update_layout(
        autosize=True,
        xaxis=dict(scaleanchor="y", scaleratio=1/aspect_ratio if aspect_ratio != 'auto' else 1),
    )
    if facet_labels:
        for i, label in enumerate(facet_labels):
            fig.layout.annotations[i]['text'] = label

    if border:
        fig.update_xaxes(showline=True, linewidth=1, linecolor='black', mirror=True)
        fig.update_yaxes(showline=True, linewidth=1, linecolor='black', mirror=True)
        
    if invert_y:
      fig.update_yaxes(
            autorange=True,
        )
        
    if subpart_border:
        # subpart_border should be a tuple of (row_start, row_end, col_start, col_end)
        for row_start, row_end, col_start, col_end in subpart_border:
            fig.add_shape(
                type="rect",
                x0=col_start - 0.5, y0=row_start - 0.5,
                x1=col_end - 0.5, y1=row_end - 0.5,
                line=dict(color="red", width=0.2),
                fillcolor="rgba(0,0,0,0)",
            )
    # Apply layout and axis updates
    fig.update_layout(**kwargs_post)
    return fig

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


def plot_mean_std(tensor, title, xaxis='Intervention Layer', yaxis='Logit Difference'):
    # Calculate mean and standard deviation along L dimension
    B, L = tensor.shape
    mean = tensor.mean(0).cpu()
    std_dev = tensor.std(0).cpu()
    print(mean.shape)
    # Create the plot
    fig = go.Figure()

    for i in range(B):
        fig.add_trace(go.Scatter(
            x=list(range(L)),
            y=tensor[i, :].cpu(),
            mode='lines',
            line=dict(color='rgba(255, 0, 0, 0.1)'),  # Blue lines with low opacity
            showlegend=False
        ))
    # Add mean line
    fig.add_trace(go.Scatter(
        x=list(range(L)),
        y=mean,
        mode='lines+markers',
        name='Mean'
    ))

    # Add standard deviation
    fig.add_trace(go.Scatter(
        x=list(range(L)),
        y=mean + std_dev,
        mode='lines',
        line=dict(width=0),
        showlegend=False
    ))

    fig.add_trace(go.Scatter(
        x=list(range(L)),
        y=mean - std_dev,
        mode='lines',
        line=dict(width=0),
        fill='tonexty',
        fillcolor='rgba(0,100,80,0.3)',
        name='Standard Deviation'
    ))

    # Customize layout
    fig.update_layout(
        title=title,
        xaxis_title=xaxis,
        yaxis_title=yaxis,
        showlegend=True
    )

    return fig

# compute_seq_ranges
def sequence_to_ranges(seq):
    if not seq:
        return []
    
    # First, sort the sequence to handle unsorted lists
    seq = sorted(set(seq))
    
    ranges = []
    start = seq[0]
    
    # Iterate through the sequence starting from the second element
    for i in range(1, len(seq)):
        # Check if the current element is not consecutive
        if seq[i] != seq[i-1] + 1:
            # Add the current range as a tuple
            ranges.append((start, seq[i-1] + 1))
            # Start a new range
            start = seq[i]
    
    # Add the last range
    ranges.append((start, seq[-1] + 1))
    return ranges
    
def to_flat_with_labels(heads, custom_layers=None):
    N_HEADS = heads.shape[2]
    if custom_layers is None:
        custom_layers = range(heads.shape[0])
    LABELS = [f"L{i}H{j}" for i in custom_layers for j in range(N_HEADS)]
    data = einops.rearrange(heads, "l p h -> (l h) p")
    return data, LABELS
  
def plot_patching_result_pos_attn(patch_res, tokenizer, all_positions, all_layers, last_n_tokens=25, title="Activation Patching"):
  last_n_tokens = 23
  prompt = patch_res["prompt"]
  patching_results = patch_res["patching_results"]
  tokens = tokenizer.tokenize(prompt[0])
  tokens = [f"({i}) " + tokens[i] for i in range(-last_n_tokens, 0)]
  seqs = sorted([last_n_tokens + p for p in all_positions])
  seqs = sequence_to_ranges(seqs)

  fig = make_subplots(rows=2, cols=2, shared_xaxes=False, shared_yaxes=False, subplot_titles=("Key", "Value", "Query",  "Output"), vertical_spacing=0.1, horizontal_spacing=0.1, row_heights=[0.2, 0.8], column_widths=[0.5, 0.5])
  k, labelsk = to_flat_with_labels(patching_results["k"][all_layers], custom_layers=all_layers)
  v, labelsv = to_flat_with_labels(patching_results["v"][all_layers], custom_layers=all_layers)
  q, labelsq = to_flat_with_labels(patching_results["q"][all_layers], custom_layers=all_layers)
  o, labelso = to_flat_with_labels(patching_results["o"][all_layers], custom_layers=all_layers)

  max_val = max(k.max(), v.max(), q.max(), o.max()).item()
  min_val = min(k.min(), v.min(), q.min(), o.min()).item()
  max_val, min_val = max(abs(max_val), abs(min_val)), -max(abs(max_val), abs(min_val))

  figk = imshow(k[:, -last_n_tokens:], y=labelsk, title="Key", zmax=max_val, zmin=min_val, x=tokens, subpart_border=[(0, len(labelsk), a, b) for a,b in seqs])
  figv = imshow(v[:, -last_n_tokens:], y=labelsv, title="Value", zmax=max_val, zmin=min_val, x=tokens, subpart_border=[(0, len(labelsv), a, b) for a,b in seqs])
  figq = imshow(q[:, -last_n_tokens:], y=labelsq, title="Query", zmax=max_val, zmin=min_val, x=tokens, subpart_border=[(0, len(labelsq), a, b) for a,b in seqs])
  figo = imshow(o[:, -last_n_tokens:], y=labelso, title="Output", zmax=max_val, zmin=min_val, x=tokens, subpart_border=[(0, len(labelso), a, b) for a,b in seqs])
  fig.layout.coloraxis = figk.layout.coloraxis
  for trace in figk.data:
      fig.add_trace(trace, row=1, col=1)

  for trace in figv.data:
      fig.add_trace(trace, row=1, col=2)
      
  for trace in figq.data:
      fig.add_trace(trace, row=2, col=1)
      
  for trace in figo.data:
      fig.add_trace(trace, row=2, col=2)

  # add shapes
  for shape in figk.layout.shapes:
      fig.add_shape(shape, row=1, col=1)
      
  for shape in figv.layout.shapes:
      fig.add_shape(shape, row=1, col=2)

  for shape in figq.layout.shapes:
      fig.add_shape(shape, row=2, col=1)

  for shape in figo.layout.shapes:
      fig.add_shape(shape, row=2, col=2)
          
  # set height    
  fig.update_layout(height=1500)

  # set y axis font size
  fig.update_yaxes(tickfont=dict(size=8))

  # add title
  fig.update_layout(title_text=title)
  return fig
  
def plot_patching_result_pos(patch_res, tokenizer, last_n_tokens=25, title="Activation Patching", filter=[], **kwargs):

  prompt = patch_res["prompt"]
  patching_results = patch_res["patching_results"]
  tokens = tokenizer.tokenize(prompt[0])
  
  keys = list(patching_results.keys())
  if len(filter) > 0:
    keys = [key for key in keys if key in filter]
    
  if "v" in keys and "k" in keys and "q" in keys and "o" in keys:
    return plot_patching_result_pos_attn(patch_res, tokenizer, last_n_tokens=last_n_tokens, title=title, **kwargs)

  # assume just blocks
  keys_to_name = {
    "attn": "Attention",
    "mlp": "MLP",
    "resid": "Residual",
    "o": "Output",
    "k": "Key",
    "v": "Value",
    "q": "Query",
  }
  data = torch.stack([patching_results[key] for key in keys])[:, :, -last_n_tokens:]
  print(data.shape)
  return imshow(data, 
              title=title,
              labels={"x": "Sequence position", "y": "Layer", "color": "Logit diff variation"},
              facet_col=0,
              x = [f"({i}) " + tokens[i] for i in range(-last_n_tokens, 0)],
              invert_y=True,
              facet_labels=[keys_to_name[key] for key in keys],
              width=2000,
              height=600,
              **kwargs
  )
  
def single_pos_attn_head_comparison(res_fs, res_ft, pos, attn_type="o", all_layers=None):
    if all_layers is None:
        all_layers = range(res_fs["patching_results"][attn_type].shape[0])
    type_to_name = {"o": "Output", "q": "Query", "k": "Key", "v": "Value"}
    data = [res_fs["patching_results"][attn_type][:, pos, :], res_ft["patching_results"][attn_type][:, pos, :]]
    fig = make_subplots(rows=1, cols=2, subplot_titles=[f"{type_to_name[attn_type]} (Few-Shot)", f"{type_to_name[attn_type]} (Finetuned)"])

    for i, d in enumerate(data):
        dfig = imshow(d, 
                title=f"Activation Patching Position {pos}",
                labels={"x": "Head", "y": "Layer", "color": "Logit diff variation"},
                x=list(range(d.shape[1])),
                invert_y=True,
                width=1000,
                subpart_border=[(a,b,0,d.shape[1]) for a,b in sequence_to_ranges(all_layers)],
                height=600,
        )
        for trace in dfig.data:
            fig.add_trace(trace, row=1, col=i+1)
        for shape in dfig.layout.shapes:
            fig.add_shape(shape, row=1, col=i+1)
            
    fig.layout.coloraxis = dfig.layout.coloraxis
    fig.update_layout(title=f"Activation Patching Query Position {pos}")
    return fig