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


## Patch Scope

import plotly.subplots as sp
import plotly.graph_objects as go
import numpy as np

MAIN_LINE = 0.7
PATCH_MAIN = 0.5
PATCH_ALT = 0.3
RECT_WIDTH= 0.05

# def get_name(head, shorten=False):
#     if head.startswith('q'):
#         return f'Attn Query{"<br>" if shorten else " "}(all heads)'
#     elif head.startswith('o'):
#         return f'Attn Output{"<br>" if shorten else " "}(all heads)'
#     elif head.startswith('m'):
#         return 'MLP'
#     else:
#         return head
    
# def add_flow_chart(fig, x_pos, layers, color, q_y_offset=0, q_x_offset=0, head='q', shorten_legend=False):
#     if head.startswith('q'):
#         symbol = 'circle'
#         x_offset = -RECT_WIDTH
#         y_offset = -2*q_y_offset
#     elif head.startswith('o'):
#         symbol = 'diamond'
#         x_offset =  -0.5*RECT_WIDTH
#         y_offset = 2*q_y_offset
#     elif head.startswith('mlp'):
#         head = head.replace('mlp', 'm')
#         symbol = 'square'
#         x_offset = 2*q_x_offset
#         y_offset = 3.5*q_y_offset
#     else:
#         raise ValueError("Invalid head type. Choose 'q', 'o' or 'mlp'.")
    
#     if not head.endswith('cross'):
#         fig.add_trace(go.Scatter(
#             x=[x_pos + x_offset/2] * len(layers),
#             y=[el + y_offset for el in layers],
#             mode='markers+text',
#             text=[head] * len(layers),
#             marker=dict(size=15, color=color, symbol=symbol, line=dict(color=color, width=2)),
#             showlegend=False,
#         ), row=1, col=1)
    
#     # input     
#     for y in layers:
#         # add rectangle (red dotted) for the patching
#         if head.startswith('m'):
#             fig.add_trace(go.Scatter(
#                 x=[MAIN_LINE, MAIN_LINE + RECT_WIDTH, MAIN_LINE + RECT_WIDTH, MAIN_LINE],
#                 y=[y+2*q_y_offset, y+2*q_y_offset, y+5*q_y_offset, y+5*q_y_offset],
#                 mode='lines',
#                 line=dict(color='red', width=1, dash='dot'),
#                 showlegend=False
#             ), row=1, col=1
#             )
#         else:
#             fig.add_trace(go.Scatter(
#                 x=[MAIN_LINE, MAIN_LINE - RECT_WIDTH, MAIN_LINE - RECT_WIDTH, MAIN_LINE],
#                 y=[y-2*q_y_offset, y-2*q_y_offset, y+2*q_y_offset, y+2*q_y_offset],
#                 mode='lines',
#                 line=dict(color='red', width=1, dash='dot'),
#                 showlegend=False
#             ), row=1, col=1)
        
#     fig.add_trace(go.Scatter(
#         x=[MAIN_LINE + x_offset] * len(layers),
#         y=[el + y_offset for el in layers],
#         mode='markers+text',
#         marker=dict(size=12, color=color, symbol=symbol, line=dict(color="black", width=1) if head.endswith('cross') else None),
#         showlegend=False,
#         name=get_name(head, shorten=shorten_legend),
#         text=[head.split("cross")[0]] * len(layers),
#     ), row=1, col=1)


#     if head.endswith('cross'):
#         fig.add_trace(go.Scatter(
#             x=[MAIN_LINE + x_offset] * len(layers),
#             y=[el + y_offset for el in layers],
#             mode='markers',
#             marker=dict(size=12, color="red", symbol="line-ne", line=dict(color="red", width=1)),
#             showlegend=False,
#             name="Avg Patch"
#         ), row=1, col=1)

#     for y in layers:
#         if not head.endswith('cross'):
#             fig.add_annotation(
#                 ax=x_pos + x_offset/2 +0.01, ay=y + y_offset,
#                 axref="x1", ayref="y1",
#                 x=MAIN_LINE + x_offset-0.02, y=y + y_offset,
#                 xref="x1", yref="y1",
#                 showarrow=True,
#                 arrowhead=2,
#                 arrowsize=1,
#                 arrowwidth=1.5,
#                 arrowcolor=color
#             )

# # Heatmaps and annotations
# def add_heatmap_annotations(fig, data, std, col, format_fun, y_values):

#     for i in range(data.shape[0]):
#         for j in range(data.shape[1]):
#             fig.add_annotation(
#                 x=j,
#                 y=y_values[i],
#                 text=str(format_fun(data[i, j].item())) + (f' <span style="font-size:10px; ">(σ={format_fun(std[i, j].item())})</span>' if std is not None else ""),
#                 showarrow=False,
#                 font=dict(color="black", size=14),
#                 xref=f"x{col}", yref=f"y{col}"
#             )

# def create_patch_scope_plot(probs, ranks, a_layers, b_layers, avg_layers, aggregation="median", a_title="Alt CTX", b_title="CTX", c_title="PRIOR", title=None, q_x_offset=0.03, q_y_offset=0.1, N_LAYERS=32, add_rank=True, add_prob=True):
#     # Create subplots for the flow chart and two heatmaps

#     probs, std_devs_probs, probs_median = probs
#     ranks, std_devs_ranks, ranks_median = ranks

#     if aggregation == "median":
#         probs = probs_median
#         ranks = ranks_median
        
#     num_rows = probs.shape[1]
    
#     num_sp = 3 if add_rank and add_prob else 2

#     sp_titles = ["Patching Flow"]
#     sp_specs = [[{'type': 'scatter'}]]
#     if add_prob:
#         sp_titles.append("Answer Likelihood")
#         sp_specs[0].append({'type': 'heatmap'})
#     if add_rank:
#         sp_titles.append("Answer Rank")
#         sp_specs[0].append({'type': 'heatmap'})
#     col_widths = [0.25 if num_rows == 3 else 0.2]
#     if add_prob:
#         col_widths.append(0.4)
#     if add_rank:
#         col_widths.append(0.4)
#     fig = sp.make_subplots(
#         rows=1, cols=num_sp,
#         shared_yaxes=False, horizontal_spacing=0.08,
#         specs=sp_specs, 
#         column_widths=col_widths,
#         subplot_titles=sp_titles
#     )

#     # add hidden scatters for legend
#     fig.add_trace(go.Scatter(
#         x=[None] ,
#         y=[None],
#         mode='markers',
#         marker=dict(size=14, color="white", symbol="square", line=dict(color="black", width=1)),
#         showlegend=True,
#         name=get_name("m", shorten=num_rows == 2),
#     ), row=1, col=1)
#     fig.add_trace(go.Scatter(
#         x=[None] ,
#         y=[None],
#         mode='markers',
#         marker=dict(size=14, color="white", symbol="circle", line=dict(color="black", width=1)),
#         showlegend=True,
#         name=get_name("q", shorten=num_rows == 2),
#     ), row=1, col=1)
#     fig.add_trace(go.Scatter(
#         x=[None] ,
#         y=[None],
#         mode='markers',
#         marker=dict(size=14, color="white", symbol="diamond", line=dict(color="black", width=1)),
#         showlegend=True,
#         name=get_name("o", shorten=num_rows == 2),
#     ), row=1, col=1)
#     fig.add_trace(go.Scatter(
#         x=[None] ,
#         y=[None],
#         mode='markers',
#             marker=dict(size=14, color="red", symbol="line-ne", line=dict(color="red", width=1)),
#         showlegend=True,
#         name="Avg Patch",
#     ), row=1, col=1)

#     # Add an annotation to mimic the text in the legend
#     if num_sp == 3 and num_rows == 3:
#         shift = 0
#         y_shift = 0.001
#     else:
#         shift = 0.007
#         y_shift = 0.0
#     if num_rows == 2:
#         y_shift = 0.01
#         shift = 0.02
#     fig.add_annotation(
#         x=0.022+shift, y=0.176+3.7*y_shift,  # Coordinates for the annotation in normalized coordinates
#         xref="paper", yref="paper",
#         text="m",  # Text you want to display
#         showarrow=False,
#         font=dict(size=12, color="black"),
#         align="left"
#     )
#     fig.add_annotation(
#         x=0.0268+shift, y=0.148+2.8*y_shift,  # Coordinates for the annotation in normalized coordinates
#         xref="paper", yref="paper",
#         text="q",  # Text you want to display
#         showarrow=False,
#         font=dict(size=12, color="black"),
#         align="left"
#     )
#     fig.add_annotation(
#         x=0.0268+shift, y=0.116+y_shift,  # Coordinates for the annotation in normalized coordinates
#         xref="paper", yref="paper",
#         text="o",  # Text you want to display
#         showarrow=False,
#         font=dict(size=12, color="black"),
#         align="left"
#     )


#     # Flow chart for PRIOR
#     flow_chart_prior = go.Scatter(
#         x=[MAIN_LINE, MAIN_LINE],
#         y=[0, N_LAYERS - 1],
#         mode='lines',
#         line=dict(color='red', width=2, dash='dot'),
#         showlegend=True,
#         name="Residual"
#     )
#     fig.add_trace(flow_chart_prior, row=1, col=1)

#     for head in a_layers:
#         add_flow_chart(fig, PATCH_ALT, a_layers[head], q_x_offset=q_x_offset, q_y_offset=q_y_offset, head=head, color="lightblue")
#     for head in b_layers:
#         add_flow_chart(fig, PATCH_MAIN, b_layers[head], q_x_offset=q_x_offset, q_y_offset=q_y_offset, head=head, color="lightgreen")
#     for head in avg_layers:
#         add_flow_chart(fig, MAIN_LINE, avg_layers[head], q_x_offset=q_x_offset, q_y_offset=q_y_offset, head=head+'cross', color="white")

    
#     if num_rows == 2:
#         axs_titles = [a_title, c_title]
#         axs_vals_flow = [PATCH_ALT, MAIN_LINE]
#         axs_vals_data = [0, 1]
#     else:
#         axs_titles = [a_title, b_title, c_title]
#         axs_vals_flow = [PATCH_ALT, PATCH_MAIN, MAIN_LINE]
#         axs_vals_data = [0, 1, 2]
#     if add_prob:
#         # Probability heatmap
#         heatmap_probs = go.Heatmap(
#             z=probs,
#             x=axs_titles,
#             y=list(range(N_LAYERS)),
#             colorscale='RdBu_r',
#             zmin=-0.2,
#             zmax=1.2,
#             showscale=False,
#         )
#         fig.add_trace(heatmap_probs, row=1, col=2)
#         add_heatmap_annotations(fig, probs, std_devs_probs, 2, lambda x: round(x, 3), list(range(N_LAYERS)))

#     if add_rank:
#         rank_col = 3 if num_sp == 3 else 2
#         # Rank heatmap
#         heatmap_ranks = go.Heatmap(
#             z=ranks,
#             x=axs_titles,
#             y=list(range(N_LAYERS)),
#             colorscale='Aggrnyl_r',
#             zmin=0,
#             zmax=500,
#             showscale=False,
#         )
#         fig.add_trace(heatmap_ranks, row=1, col=rank_col)
#         add_heatmap_annotations(fig, ranks, std_devs_ranks, rank_col, lambda x: int(x), list(range(N_LAYERS)))

#     # Adjust layout for the inverted y-axis
#     fig.update_yaxes(
#         tickvals=list(range(N_LAYERS)),
#         ticktext=list(range(N_LAYERS)),  # Ensure tick labels match the inverted y-axis
#         showgrid=True,
#         ticks="outside",
#         ticklen=6,
#         minor_ticks="outside",
#         tickwidth=1
#     )
    
#     fig.update_yaxes(
#         title_text="Layer Number",
#         row = 1, col = 1,
#     )

#     fig.update_xaxes(
#         title_text="Patching Sources",
#         tickvals=axs_vals_flow,
#         ticktext=axs_titles,
#         range=[0.2, 0.8],
#         showline=True,
#         linewidth=1,
#         linecolor='black',
#         ticks="outside",
#         ticklen=6,
#         row=1, col=1
#     )

#     fig.update_xaxes(
#         title_text="Answer Token",
#         tickvals=axs_vals_data,
#         ticktext=axs_titles,
#         showgrid=True,
#         ticks="outside",
#         ticklen=6,
#         minor_ticks="outside",
#         tickwidth=1,
#         row=1, col=2
#     )
#     if num_sp == 3:
#         fig.update_xaxes(
#             title_text="Answer Token",
#             tickvals=axs_vals_data,
#             ticktext=axs_titles,
#             showgrid=True,
#             ticks="outside",
#             ticklen=6,
#             minor_ticks="outside",
#             tickwidth=1,
#             row=1, col=3
#     )

#     if num_sp == 2:
#         width= 800
#     elif num_sp == 3:
#         width = 1200
    
#     if num_rows == 2:
#         width-=300
        
#     # Add borders around the subplots
#     margin_top = 40 if title is None else 100
#     fig.update_layout(
#         height=800,
#         width=width,
#         title_text=title,
#         margin=dict(l=40, r=40, t=margin_top, b=40),
#         plot_bgcolor='rgba(0,0,0,0)',
#         xaxis2=dict(showline=True, linewidth=1, linecolor='black', mirror=True),
#         yaxis2=dict(showline=True, linewidth=1, linecolor='black', mirror=True),
#         xaxis3=dict(showline=True, linewidth=1, linecolor='black', mirror=True),
#         yaxis3=dict(showline=True, linewidth=1, linecolor='black', mirror=True),
#     )

#     fig.update_yaxes(range=[0, N_LAYERS - 1], row=1, col=1)

#     # Define padding for the subplot 1
#     padding = 0.5
#     fig.update_layout(
#         yaxis1=dict(range=[0-padding, N_LAYERS-1+padding]),
#         legend=dict(
#             x=0.01,  
#             y=0.05,  # Position the legend above the plot
#             orientation="v",  # Horizontal orientation
#             xanchor="left",  # Align the legend center with the x position
#             yanchor="bottom",  # Align the bottom of the legend with the y position,
#             # border
#             bordercolor="black",  # Color of the border
#             borderwidth=1,  # Width of the border
#             bgcolor="white"  # Background color of the legend
#         ),
#     )
#     return fig


# def create_patch_scope_lplot(probs, aggregation, ranks, a_layers, b_layers, avg_layers, a_title="Alt CTX", b_title="CTX", c_title="PRIOR", title=None, q_x_offset=0.03, q_y_offset=0.1, N_LAYERS=32, add_rank=True, add_prob=True):
#     # Create subplots for the flow chart and two heatmaps

#     probs, std_devs, probs_median = probs
#     ranks, std_devs_ranks, ranks_median = ranks
    
#     num_rows = probs.shape[1]
    
#     num_sp = 3 if add_rank and add_prob else 2
#     sp_titles = ["Patching Flow"]
#     sp_specs = [[{'type': 'scatter'}]]
#     if add_prob:
#         sp_titles.append("Answer Likelihood")
#         sp_specs[0].append({'type': 'heatmap'})
#     if add_rank:
#         sp_titles.append("Answer Rank")
#         sp_specs[0].append({'type': 'heatmap'})
#     col_widths = [0.25 if num_rows == 3 else 0.2]
#     if add_prob:
#         col_widths.append(0.4)
#     if add_rank:
#         col_widths.append(0.4)
#     fig = sp.make_subplots(
#         rows=1, cols=num_sp,
#         shared_yaxes=False, horizontal_spacing=0.18,
#         specs=sp_specs, 
#         column_widths=col_widths,
#         subplot_titles=sp_titles
#     )

#     # add hidden scatters for legend
#     fig.add_trace(go.Scatter(
#         x=[None] ,
#         y=[None],
#         mode='markers',
#         marker=dict(size=14, color="white", symbol="square", line=dict(color="black", width=1)),
#         showlegend=True,
#         legend="legend2",
#         name=get_name("m", shorten=num_rows == 2),
#     ), row=1, col=1)
#     fig.add_trace(go.Scatter(
#         x=[None] ,
#         y=[None],
#         mode='markers',
#         marker=dict(size=14, color="white", symbol="circle", line=dict(color="black", width=1)),
#         showlegend=True,
#         legend="legend2",

#         name=get_name("q", shorten=num_rows == 2),
#     ), row=1, col=1)
#     fig.add_trace(go.Scatter(
#         x=[None] ,
#         y=[None],
#         mode='markers',
#         marker=dict(size=14, color="white", symbol="diamond", line=dict(color="black", width=1)),
#         showlegend=True,
#         legend="legend2",

#         name=get_name("o", shorten=num_rows == 2),
#     ), row=1, col=1)
#     fig.add_trace(go.Scatter(
#         x=[None] ,
#         y=[None],
#         mode='markers',
#             marker=dict(size=14, color="red", symbol="line-ne", line=dict(color="red", width=1)),
#         showlegend=True,
#         legend="legend2",

#         name="Avg Patch",
#     ), row=1, col=1)
    
    


#     # Add an annotation to mimic the text in the legend
#     if num_sp == 3 and num_rows == 3:
#         shift = 0
#         y_shift = 0
#     else:
#         shift = 0.007
#         y_shift = 0.0
#     if num_rows == 2:
#         y_shift = 0.01
#         shift = 0.02
#     # fig.add_annotation(
#     #     x=0.022+shift, y=0.176+3.7*y_shift,  # Coordinates for the annotation in normalized coordinates
#     #     xref="paper", yref="paper",
#     #     text="m",  # Text you want to display
#     #     showarrow=False,
#     #     font=dict(size=12, color="black"),
#     #     align="center"
#     # )
#     # fig.add_annotation(
#     #     x=0.0268+shift, y=0.148+2.8*y_shift,  # Coordinates for the annotation in normalized coordinates
#     #     xref="paper", yref="paper",
#     #     text="q",  # Text you want to display
#     #     showarrow=False,
#     #     font=dict(size=12, color="black"),
#     #     align="left"
#     # )
#     # fig.add_annotation(
#     #     x=0.0268+shift, y=0.116+y_shift,  # Coordinates for the annotation in normalized coordinates
#     #     xref="paper", yref="paper",
#     #     text="o",  # Text you want to display
#     #     showarrow=False,
#     #     font=dict(size=12, color="black"),
#     #     align="left"
#     # )


#     # Flow chart for PRIOR
#     flow_chart_prior = go.Scatter(
#         x=[MAIN_LINE, MAIN_LINE],
#         y=[0, N_LAYERS - 1],
#         mode='lines',
#         line=dict(color='red', width=2, dash='dot'),
#         showlegend=True,
#         name="Residual"
#     )
#     fig.add_trace(flow_chart_prior, row=1, col=1)

#     for head in a_layers:
#         add_flow_chart(fig, PATCH_ALT, a_layers[head], q_x_offset=q_x_offset, q_y_offset=q_y_offset, head=head, color="lightblue")
#     for head in b_layers:
#         add_flow_chart(fig, PATCH_MAIN, b_layers[head], q_x_offset=q_x_offset, q_y_offset=q_y_offset, head=head, color="lightgreen")
#     for head in avg_layers:
#         add_flow_chart(fig, MAIN_LINE, avg_layers[head], q_x_offset=q_x_offset, q_y_offset=q_y_offset, head=head+'cross', color="white")

    
#     if num_rows == 2:
#         axs_titles = [a_title, c_title]
#         axs_vals_flow = [PATCH_ALT, MAIN_LINE]
#         axs_vals_data = [0, 1]
#     else:
#         axs_titles = [a_title, b_title, c_title]
#         axs_vals_flow = [PATCH_ALT, PATCH_MAIN, MAIN_LINE]
#         axs_vals_data = [0, 1, 2]
#     if add_prob:        
#         # Creating shaded areas for each line
#         upper = probs + std_devs
#         # bound to 1
#         upper[upper > 1] = 1
        
#         lower = probs - std_devs
#         # bound to 0
#         lower[lower < 0] = 0
#         line1_upper = go.Scatter(
#             x=upper[:, 0],
#             y=list(range(N_LAYERS)),
#             mode='lines',
#             line=dict(width=0),
#             fill=None,
#             showlegend=False,
#         )

#         line1_lower = go.Scatter(
#             x=lower[:, 0],
#             y=list(range(N_LAYERS)),
#             mode='lines',
#             line=dict(width=0),
#             fill='tonextx',
#             fillcolor='rgba(0,100,80,0.2)',
#             showlegend=False
#         )

#         line1 = go.Scatter(
#             x=probs[:, 0],
#             y=list(range(N_LAYERS)),
#             mode='lines',
#             name=axs_titles[0],
#             legend='legend2',

#             # color
#             line=dict(color='rgba(0,100,80,1.0)')
#         )

#         # median (dotted)
#         median1 = go.Scatter(
#             x=probs_median[:, 0],
#             y=list(range(N_LAYERS)),
#             mode='lines',
#             line=dict(width=1, dash='dot', color = 'rgba(0,100,80,1.0)'),
#             name='Median',
#             showlegend=False,
            
#         )

#         line2_upper = go.Scatter(
#             x=upper[:, 1],
#             y=list(range(N_LAYERS)),
#             mode='lines',
#             line=dict(width=0),
#             fill=None,
#             showlegend=False
#         )

#         line2_lower = go.Scatter(
#             x=lower[:, 1],
#             y=list(range(N_LAYERS)),
#             mode='lines',
#             line=dict(width=0),
#             fill='tonextx',
#             fillcolor='rgba(100,0,80,0.2)',
#             showlegend=False
#         )

#         line2 = go.Scatter(
#             x=probs[:, 1],
#             y=list(range(N_LAYERS)),
#             mode='lines',
#             name=axs_titles[1],
#             legend='legend2',
#             # color
#             line=dict(color='rgba(100,0,80,1.0)')
            
#         )
        
#         median2 = go.Scatter(
#             x=probs_median[:, 1],
#             y=list(range(N_LAYERS)),
#             mode='lines',
#             line=dict(width=1, dash='dot', color = 'rgba(100,0,80,1.0)'),
#             name='Median',
#             showlegend=False,
#         )
        

#         fig.add_trace(line1_upper, row=1, col=2)
#         fig.add_trace(line1_lower, row=1, col=2)
#         fig.add_trace(line1, row=1, col=2)
#         fig.add_trace(median1, row=1, col=2)
#         fig.add_trace(line2_upper, row=1, col=2)
#         fig.add_trace(line2_lower, row=1, col=2)
#         fig.add_trace(line2, row=1, col=2)
#         fig.add_trace(median2, row=1, col=2)
        
#         if num_rows == 3:
#             upper = probs + std_devs
#             # bound to 1
#             upper[upper > 1] = 1

#             lower = probs - std_devs
#             # bound to 0
#             lower[lower < 0] = 0
#             line3_upper = go.Scatter(
#                 x=upper[:, 2],
#                 y=list(range(N_LAYERS)),
#                 mode='lines',
#                 line=dict(width=0),
#                 fill=None,
#                 showlegend=False,
#             )

#             line3_lower = go.Scatter(
#                 x=lower[:, 2],
#                 y=list(range(N_LAYERS)),
#                 mode='lines',
#                 line=dict(width=0),
#                 fill='tonextx',
#                 fillcolor='rgba(0,100,80,0.2)',
#                 showlegend=False
#             )

#             line3 = go.Scatter(
#                 x=probs[:, 2],
#                 y=list(range(N_LAYERS)),
#                 mode='lines',
#                 name=axs_titles[2],
#                 legend='legend2',
#                 # color
#                 line=dict(color='rgba(0,0,0,1.0)')
#             )

#             # median (dotted)
#             median3 = go.Scatter(
#                 x=probs_median[:, 2],
#                 y=list(range(N_LAYERS)),
#                 mode='lines',
#                 line=dict(width=1, dash='dot', color = 'rgba(0,0,0,1.0)'),
#                 name='Median',
#                 showlegend=False,

#             )

#             fig.add_trace(line3_upper, row=1, col=2)
#             fig.add_trace(line3_lower, row=1, col=2)
#             fig.add_trace(line3, row=1, col=2)
#             fig.add_trace(median3, row=1, col=2)
            
#     # legend entries for median and mean
#     fig.add_trace(go.Scatter(
#         x=[None] ,
#         y=[None],
#         mode='lines',
#         showlegend=True,
#         legend='legend2',
#         line=dict(width=1,color = 'black'),
#         name="Mean",
#     ), row=1, col=2)

#     fig.add_trace(go.Scatter(
#         x=[None] ,
#         y=[None],
#         mode='lines',
#         showlegend=True,
#         legend='legend2',
#         line=dict(width=1, dash='dot', color = 'black'),
#         name="Median",
#     ), row=1, col=2)

#     if add_rank:
#         # Rank heatmap
#         heatmap_ranks = go.Heatmap(
#             z=ranks[0],
#             x=axs_titles,
#             y=list(range(N_LAYERS)),
#             colorscale='Aggrnyl_r',
#             zmin=0,
#             zmax=500,
#             showscale=False,
#         )
#         fig.add_trace(heatmap_ranks, row=1, col=3)
#         add_heatmap_annotations(fig, ranks, 3, lambda x: int(x), list(range(N_LAYERS)))

#     # Adjust layout for the inverted y-axis
#     fig.update_yaxes(
#         tickvals=list(range(N_LAYERS)),
#         ticktext=list(range(N_LAYERS)),  # Ensure tick labels match the inverted y-axis
#         showgrid=True,
#         ticks="outside",
#         ticklen=6,
#         minor_ticks="outside",
#         tickwidth=1
#     )
    
#     fig.update_yaxes(
#         title_text="Layer Number",
#         row = 1, col = 1,
#     )

#     fig.update_xaxes(
#         title_text="Patching Sources",
#         tickvals=axs_vals_flow,
#         ticktext=axs_titles,
#         range=[0.2, 0.8],
#         showline=True,
#         linewidth=1,
#         linecolor='black',
#         ticks="outside",
#         ticklen=6,
#         row=1, col=1
#     )

#     fig.update_xaxes(
#         title_text="Likelihood",
#         showgrid=True,
#         ticks="outside",
#         ticklen=6,
#         minor_ticks="outside",
#         tickwidth=1,
#         row=1, col=2
#     )
#     if num_sp == 3:
#         fig.update_xaxes(
#             title_text="Answer Token",
#             tickvals=axs_vals_data,
#             ticktext=axs_titles,
#             showgrid=True,
#             ticks="outside",
#             ticklen=6,
#             minor_ticks="outside",
#             tickwidth=1,
#             row=1, col=3
#     )

#     if num_sp == 2:
#         width= 800
#     elif num_sp == 3:
#         width = 1200
    
#     if num_rows == 2:
#         width-=300
        
#     # Add borders around the subplots
#     margin_top = 40 if title is None else 100
#     fig.update_layout(
#         height=800,
#         width=width,
#         title_text=title,
#         margin=dict(l=40, r=40, t=margin_top, b=40),
#         plot_bgcolor='rgba(0,0,0,0)',
#         xaxis2=dict(showline=True, linewidth=1, linecolor='black', mirror=True),
#         yaxis2=dict(showline=True, linewidth=1, linecolor='black', mirror=True),
#         xaxis3=dict(showline=True, linewidth=1, linecolor='black', mirror=True),
#         yaxis3=dict(showline=True, linewidth=1, linecolor='black', mirror=True),
#     )

#     fig.update_yaxes(range=[0, N_LAYERS - 1], row=1, col=1)
#     fig.update_yaxes(range=[0, N_LAYERS - 1], row=1, col=2)

#     # Define padding for the subplot 1
#     padding = 0.5
#     fig.update_layout(
#         yaxis1=dict(range=[0-padding, N_LAYERS-1+padding]),
#         legend1=dict(
#             x=0.01,  
#             y=0.05,  # Position the legend above the plot
#             orientation="v",  # Horizontal orientation
#             xanchor="left",  # Align the legend center with the x position
#             yanchor="bottom",  # Align the bottom of the legend with the y position,
#             # border
#             bordercolor="black",  # Color of the border
#             borderwidth=1,  # Width of the border
#             bgcolor="white"  # Background color of the legend
#         ),
#         legend2=dict(
#             x=0.51,  
#             y=0.05,  # Position the legend above the plot
#             orientation="v",  # Horizontal orientation
#             xanchor="left",  # Align the legend center with the x position
#             yanchor="bottom",  # Align the bottom of the legend with the y position,
#             # border
#             bordercolor="black",  # Color of the border
#             borderwidth=1,  # Width of the border
#             bgcolor="white"  # Background color of the legend
#         ),
#     )
#     return fig



def get_name(head, shorten=False):
    """Returns a formatted name based on the head type."""
    if head.startswith('q'):
        return f'Attn Query{"<br>" if shorten else " "}(all heads)'
    elif head.startswith('o'):
        return f'Attn Output{"<br>" if shorten else " "}(all heads)'
    elif head.startswith('m'):
        return 'MLP'
    else:
        return head

def get_flow_chart_params(head, q_y_offset, q_x_offset):
    """Returns the symbol, x_offset, and y_offset based on the head type."""
    if head.startswith('q'):
        return 'circle', -RECT_WIDTH, -2 * q_y_offset, head
    elif head.startswith('o'):
        return 'diamond', -0.5 * RECT_WIDTH, 2 * q_y_offset, head
    elif head.startswith('mlp'):
        head = head.replace('mlp', 'm')
        return 'square', 2 * q_x_offset, 3.5 * q_y_offset, head
    else:
        raise ValueError("Invalid head type. Choose 'q', 'o' or 'mlp'.")

def add_rectangle(fig, head, y, q_y_offset):
    """Adds a rectangle (red dotted) for patching."""
    x_vals = [MAIN_LINE, MAIN_LINE + RECT_WIDTH, MAIN_LINE + RECT_WIDTH, MAIN_LINE] if head.startswith('m') else [MAIN_LINE, MAIN_LINE - RECT_WIDTH, MAIN_LINE - RECT_WIDTH, MAIN_LINE]
    y_vals = [y + 2 * q_y_offset, y + 2 * q_y_offset, y + 5 * q_y_offset, y + 5 * q_y_offset] if head.startswith('m') else [y - 2 * q_y_offset, y - 2 * q_y_offset, y + 2 * q_y_offset, y + 2 * q_y_offset]
    fig.add_trace(go.Scatter(
        x=x_vals,
        y=y_vals,
        mode='lines',
        line=dict(color='red', width=1, dash='dot'),
        showlegend=False
    ), row=1, col=1)

def add_marker_trace(fig, x_pos, y_vals, head, color, symbol, legend=False, shorten=False):
    """Adds a scatter trace with markers and text."""
    fig.add_trace(go.Scatter(
        x=[x_pos] * len(y_vals),
        y=y_vals,
        mode='markers+text',
        text=[head.split("cross")[0]] * len(y_vals),
        marker=dict(size=12, color=color, symbol=symbol, line=dict(color="black", width=1) if head.endswith('cross') else None),
        showlegend=legend,
        name=get_name(head, shorten=shorten)
    ), row=1, col=1)

def add_flow_chart(fig, x_pos, layers, color, q_y_offset=0, q_x_offset=0, head='q', shorten_legend=False):
    """Adds a flow chart to the figure."""
    symbol, x_offset, y_offset, head = get_flow_chart_params(head, q_y_offset, q_x_offset)

    if not head.endswith('cross'):
        add_marker_trace(fig, x_pos + x_offset/2, [el + y_offset for el in layers], head, color, symbol)
    
    for y in layers:
        add_rectangle(fig, head, y, q_y_offset)
        
    add_marker_trace(fig, MAIN_LINE + x_offset, [el + y_offset for el in layers], head, color, symbol)

    if head.endswith('cross'):
        fig.add_trace(go.Scatter(
            x=[MAIN_LINE + x_offset] * len(layers),
            y=[el + y_offset for el in layers],
            mode='markers',
            marker=dict(size=12, color="red", symbol="line-ne", line=dict(color="red", width=1)),
            showlegend=False,
            name="Avg Patch"
        ), row=1, col=1)

    for y in layers:
        if not head.endswith('cross'):
            fig.add_annotation(
                ax=x_pos + x_offset/2 + 0.01, ay=y + y_offset,
                axref="x1", ayref="y1",
                x=MAIN_LINE + x_offset-0.02, y=y + y_offset,
                xref="x1", yref="y1",
                showarrow=True,
                arrowhead=2,
                arrowsize=1,
                arrowwidth=1.5,
                arrowcolor=color
            )

def add_heatmap_annotations(fig, data, std, col, format_fun, y_values):
    """Adds annotations to heatmaps."""
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            fig.add_annotation(
                x=j,
                y=y_values[i],
                text=f"{format_fun(data[i, j].item())}" + (f' <span style="font-size:10px;">(σ={format_fun(std[i, j].item())})</span>' if std is not None else ""),
                showarrow=False,
                font=dict(color="black", size=14),
                xref=f"x{col}", yref=f"y{col}"
            )

def create_patch_scope_plot(probs, ranks, a_layers, b_layers, avg_layers, aggregation="median", a_title="Alt CTX", b_title="CTX", c_title="PRIOR", title=None, q_x_offset=0.03, q_y_offset=0.1, N_LAYERS=32, add_rank=True, add_prob=True):
    """Creates a patch scope plot with flow charts and heatmaps."""
    probs, std_devs_probs, probs_median = probs
    ranks, std_devs_ranks, ranks_median = ranks

    if aggregation == "median":
        probs = probs_median
        ranks = ranks_median

    num_sp = 3 if add_rank and add_prob else 2
    num_rows = probs.shape[1]
    col_widths = [0.25 if num_rows == 3 else 0.2] + [0.4] * (num_sp - 1)

    fig = sp.make_subplots(
        rows=1, cols=num_sp,
        shared_yaxes=False, horizontal_spacing=0.08,
        specs=[[{'type': 'scatter'}] + [{'type': 'heatmap'}] * (num_sp - 1)],
        column_widths=col_widths,
        subplot_titles=["<b>Patching Flow</b>"] + (["<b>Answer Likelihood</b>"] if add_prob else []) + (["Answer Rank"] if add_rank else []),
    )

    # Add legend for flow chart symbols
    for head, symbol, color in zip(['m', 'q', 'o'], ["square", "circle", "diamond"], ["white"] * 3):
        fig.add_trace(go.Scatter(
            x=[None],
            y=[None],
            mode='markers',
            marker=dict(size=14, color=color, symbol=symbol, line=dict(color="black", width=1)),
            showlegend=True,
            name=get_name(head, shorten=num_rows == 2),
        ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=[None],
        y=[None],
        mode='markers',
        marker=dict(size=14, color="red", symbol="line-ne", line=dict(color="red", width=1)),
        showlegend=True,
        name="Avg Patch",
    ), row=1, col=1)

    # Add flow chart for PRIOR
    fig.add_trace(go.Scatter(
        x=[MAIN_LINE, MAIN_LINE],
        y=[0, N_LAYERS - 1],
        mode='lines',
        line=dict(color='red', width=2, dash='dot'),
        showlegend=True,
        name="Residual"
    ), row=1, col=1)

    for head in a_layers:
        add_flow_chart(fig, PATCH_ALT, a_layers[head], q_x_offset=q_x_offset, q_y_offset=q_y_offset, head=head, color="lightblue")
    for head in b_layers:
        add_flow_chart(fig, PATCH_MAIN, b_layers[head], q_x_offset=q_x_offset, q_y_offset=q_y_offset, head=head, color="lightgreen")
    for head in avg_layers:
        add_flow_chart(fig, MAIN_LINE, avg_layers[head], q_x_offset=q_x_offset, q_y_offset=q_y_offset, head=head+'cross', color="white")

    axs_titles = [a_title, c_title] if num_rows == 2 else [a_title, b_title, c_title]
    axs_vals_flow = [PATCH_ALT, MAIN_LINE] if num_rows == 2 else [PATCH_ALT, PATCH_MAIN, MAIN_LINE]
    axs_vals_data = [0, 1] if num_rows == 2 else [0, 1, 2]

    if add_prob:
        # Probability heatmap
        fig.add_trace(go.Heatmap(
            z=probs,
            x=axs_titles,
            y=list(range(N_LAYERS)),
            colorscale='RdBu_r',
            zmin=-0.2,
            zmax=1.2,
            showscale=False,
        ), row=1, col=2)
        add_heatmap_annotations(fig, probs, std_devs_probs, 2, lambda x: round(x, 3), list(range(N_LAYERS)))

    if add_rank:
        # Rank heatmap
        rank_col = 3 if num_sp == 3 else 2
        fig.add_trace(go.Heatmap(
            z=ranks,
            x=axs_titles,
            y=list(range(N_LAYERS)),
            colorscale='Aggrnyl_r',
            zmin=0,
            zmax=500,
            showscale=False,
        ), row=1, col=rank_col)
        add_heatmap_annotations(fig, ranks, std_devs_ranks, rank_col, lambda x: int(x), list(range(N_LAYERS)))

    # Adjust layout for the inverted y-axis and other settings
    fig.update_yaxes(tickvals=list(range(N_LAYERS)), ticktext=list(range(N_LAYERS)), showgrid=True, ticks="outside", ticklen=6, minor_ticks="outside", tickwidth=1)
    fig.update_xaxes(title_text="Patching Sources", tickvals=axs_vals_flow, ticktext=axs_titles, range=[0.2, 0.8], showline=True, linewidth=1, linecolor='black', ticks="outside", ticklen=6, row=1, col=1)
    fig.update_xaxes(title_text="Answer Token", tickvals=axs_vals_data, ticktext=axs_titles, showgrid=True, ticks="outside", ticklen=6, minor_ticks="outside", tickwidth=1, row=1, col=2)
    if num_sp == 3:
        fig.update_xaxes(title_text="Answer Token", tickvals=axs_vals_data, ticktext=axs_titles, showgrid=True, ticks="outside", ticklen=6, minor_ticks="outside", tickwidth=1, row=1, col=3)

    # Adjust layout dimensions and margins
    width = 600 if num_sp == 2 else 1200
    width -= 300 if num_rows == 2 else 0
    fig.update_layout(height=800, width=width, title_text=title, margin=dict(l=40, r=40, t=100 if title else 40, b=40), plot_bgcolor='rgba(0,0,0,0)')
    fig.update_yaxes(range=[0, N_LAYERS - 1], row=1, col=1)
    fig.update_layout(
        yaxis1=dict(range=[-0.5, N_LAYERS-1+0.5]),
        legend=dict(
            x=0.01, y=0.05,
            orientation="v",
            xanchor="left", yanchor="bottom",
            bordercolor="black", borderwidth=1, bgcolor="white"
        )
    )

    return fig


def get_flow_chart_params_l(head, q_y_offset, q_x_offset):
    """Returns the symbol, x_offset, and y_offset based on the head type."""
    if head.startswith('q'):
        return 'circle', -RECT_WIDTH, -2 * q_y_offset
    elif head.startswith('o'):
        return 'diamond', -0.5 * RECT_WIDTH, 2 * q_y_offset
    elif head.startswith('mlp'):
        head = head.replace('mlp', 'm')
        return 'square', 2 * q_x_offset, 3.5 * q_y_offset
    else:
        raise ValueError("Invalid head type. Choose 'q', 'o' or 'mlp'.")

def add_rectangle_l(fig, head, y, q_y_offset, color):
    """Adds a rectangle (red dotted) for patching."""
    x_vals = [y + 2 * q_y_offset, y + 2 * q_y_offset, y + 5 * q_y_offset, y + 5 * q_y_offset] if head.startswith('m') else [y - 2 * q_y_offset, y - 2 * q_y_offset, y + 2 * q_y_offset, y + 2 * q_y_offset]
    y_vals = [MAIN_LINE, MAIN_LINE + RECT_WIDTH, MAIN_LINE + RECT_WIDTH, MAIN_LINE] if head.startswith('m') else [MAIN_LINE, MAIN_LINE - RECT_WIDTH, MAIN_LINE - RECT_WIDTH, MAIN_LINE]
    fig.add_trace(go.Scatter(
        x=x_vals,
        y=y_vals,
        mode='lines',
        line=dict(color=color, width=1, dash='dot'),
        showlegend=False
    ), row=1, col=1)

def add_marker_trace_l(fig, x_vals, y_pos, head, color, symbol, legend=False, shorten=False):
    """Adds a scatter trace with markers and text."""
    fig.add_trace(go.Scatter(
        x=x_vals,
        y=[y_pos] * len(x_vals),
        mode='markers+text',
        text=[head.split("cross")[0]] * len(x_vals),
        marker=dict(size=12, color=color, symbol=symbol, line=dict(color="black", width=1) if head.endswith('cross') else None),
        showlegend=legend,
        name=get_name(head, shorten=False)
    ), row=1, col=1)

def add_flow_chart_l(fig, y_pos, layers, color, residual_color, q_y_offset=0, q_x_offset=0, head='q', shorten_legend=False):
    """Adds a flow chart to the figure."""
    symbol, x_offset, y_offset, head = get_flow_chart_params(head, q_y_offset, q_x_offset)

    if not head.endswith('cross'):
        add_marker_trace_l(fig, [el + y_offset for el in layers], y_pos + x_offset/2, head, color, symbol)
    
    for x in layers:
        add_rectangle_l(fig, head, x, q_y_offset, residual_color)
        
    add_marker_trace_l(fig, [el + y_offset for el in layers], MAIN_LINE + x_offset, head, color, symbol)

    if head.endswith('cross'):
        fig.add_trace(go.Scatter(
            x=[el + y_offset for el in layers],
            y=[MAIN_LINE + x_offset] * len(layers),
            mode='markers',
            marker=dict(size=12, color="red", symbol="line-ne", line=dict(color="red", width=1)),
            showlegend=False,
            name="Avg Patch"
        ), row=1, col=1)

    for x in layers:
        if not head.endswith('cross'):
            fig.add_annotation(
                ax=x + y_offset, ay=y_pos + x_offset/2 + 0.01,
                axref="x1", ayref="y1",
                x=x + y_offset, y=MAIN_LINE + x_offset - 0.02,
                xref="x1", yref="y1",
                showarrow=True,
                arrowhead=2,
                arrowsize=1,
                arrowwidth=1.5,
                arrowcolor=color
            )
            
def create_patch_scope_lplot(probs, ranks, a_layers, b_layers, avg_layers, aggregation="median", a_title="Alt CTX", b_title="CTX", c_title="PRIOR", title=None, q_x_offset=0.03, q_y_offset=0.1, N_LAYERS=32, add_rank=True, add_prob=True):
    """Creates a patch scope plot with flow charts and line plots."""
    probs, std_devs_probs, probs_median = probs
    ranks, std_devs_ranks, ranks_median = ranks

    num_sp = 3 if add_rank and add_prob else 2
    num_rows = probs.shape[1]
    colors = ['rgba(31, 119, 180, {})', 'rgba(255, 127, 14, {})','rgba(44, 160, 44, {})']

    fig = sp.make_subplots(
        rows=num_sp, cols=1,  # Adjusted to have rows instead of columns
        shared_xaxes=True, vertical_spacing=0.15,
        specs=[[{'type': 'scatter'}]] + [[{'type': 'scatter'}] for _ in range(num_sp-1)],
        row_heights=[0.4, 0.6] 
    )

    # Add flow chart  
    fig.add_trace(go.Scatter(
        x=[0, N_LAYERS - 1],
        y=[MAIN_LINE, MAIN_LINE],
        mode='lines',
        line=dict(color=colors[num_rows-1].format(1.0), width=2, dash='dot'),
        showlegend=True,
        legend="legend1",
        name="Residual"
    ), row=1, col=1)

    for head in a_layers:
        add_flow_chart_l(fig, PATCH_ALT, a_layers[head], q_x_offset=q_x_offset, q_y_offset=q_y_offset, head=head, color=colors[0].format(0.4), residual_color=colors[1].format(1.0))
    for head in b_layers:
        add_flow_chart_l(fig, PATCH_MAIN, b_layers[head], q_x_offset=q_x_offset, q_y_offset=q_y_offset, head=head, color=colors[1].format(0.4), residual_color=colors[1].format(1.0))
    for head in avg_layers:
        add_flow_chart_l(fig, MAIN_LINE, avg_layers[head], q_x_offset=q_x_offset, q_y_offset=q_y_offset, head=head+'cross', color="white", residual_color=colors[1].format(1.0))


    axs_titles = [a_title, c_title] if num_rows == 2 else [a_title, b_title, c_title]
    axs_vals_flow = [PATCH_ALT, MAIN_LINE] if num_rows == 2 else [PATCH_ALT, PATCH_MAIN, MAIN_LINE]
    axs_vals_data = [0, 1] if num_rows == 2 else [0, 1, 2]
    next_row = 2
    
    # Add legend for flow chart symbols
    for head, symbol, color in zip(['m', 'q', 'o'], ["square", "circle", "diamond"], ["white"] * 3):
        if head in a_layers:
            fig.add_trace(go.Scatter(
                x=[None],
                y=[None],
                mode='markers',
                marker=dict(size=14, color=color, symbol=symbol, line=dict(color="black", width=1)),
                showlegend=True,
                name=get_name(head, shorten=False),
                legend="legend1",
            ), row=1, col=1)
        # fig.add_trace(go.Scatter(
        #     x=[None],
        #     y=[None],
        #     mode='markers',
        #     marker=dict(size=14, color=color, symbol=symbol, line=dict(color="black", width=1)),
        #     showlegend=True,
        #     name=get_name(head, shorten=False),
        #     legend="legend1",
        # ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=[None],
        y=[None],
        mode='markers',
        marker=dict(size=14, color="red", symbol="line-ne", line=dict(color="red", width=1)),
        showlegend=True,
        name="Avg Patch",
        legend="legend1",
    ), row=1, col=1)
    
    # Probability line plot
    if add_prob:
        upper = probs + std_devs_probs
        upper[upper > 1] = 1
        lower = probs - std_devs_probs
        lower[lower < 0] = 0

        for i in range(num_rows):
            line_upper = go.Scatter(
                x=list(range(N_LAYERS)),
                y=upper[:, i],
                mode='lines',
                line=dict(width=0),
                fill=None,
                showlegend=False,
            )

            line_lower = go.Scatter(
                x=list(range(N_LAYERS)),
                y=lower[:, i],
                mode='lines',
                line=dict(width=0),
                fill='tonexty',
                fillcolor=colors[i].format(0.2),
                showlegend=False
            )

            line_mean = go.Scatter(
                x=list(range(N_LAYERS)),
                y=probs[:, i],
                mode='lines',
                name=axs_titles[i],
                line=dict(color=colors[i].format(1.0)),
                legend="legend2",
                legendgroup="values"
            )

            line_median = go.Scatter(
                x=list(range(N_LAYERS)),
                y=probs_median[:, i],
                mode='lines',
                line=dict(width=1, dash='dot', color=colors[i].format(1.0)),
                name='Median',
                showlegend=False
            )

            fig.add_trace(line_upper, row=next_row, col=1)
            fig.add_trace(line_lower, row=next_row, col=1)
            fig.add_trace(line_mean, row=next_row, col=1)
            fig.add_trace(line_median, row=next_row, col=1)
        # legend entries for median and mean
        fig.add_trace(go.Scatter(
            x=[None] ,
            y=[None],
            mode='lines',
            showlegend=True,
            legend='legend2',
            line=dict(width=1,color = 'black'),
            name="Mean",
            legendgroup="meta"

        ), row=next_row, col=1)

        fig.add_trace(go.Scatter(
            x=[None] ,
            y=[None],
            mode='lines',
            showlegend=True,
            legend='legend2',
            line=dict(width=1, dash='dot', color = 'black'),
            name="Median",
            legendgroup="meta"

        ), row=next_row, col=1)
        
        next_row += 1

    # Rank line plot
    if add_rank:
        rank_upper = ranks + std_devs_ranks
        rank_upper[rank_upper > 500] = 500
        rank_lower = ranks - std_devs_ranks
        rank_lower[rank_lower < 0] = 0

        for i in range(num_rows):
            rank_line_upper = go.Scatter(
                x=list(range(N_LAYERS)),
                y=rank_upper[:, i],
                mode='lines',
                line=dict(width=0),
                fill=None,
                showlegend=False,
            )

            rank_line_lower = go.Scatter(
                x=list(range(N_LAYERS)),
                y=rank_lower[:, i],
                mode='lines',
                line=dict(width=0),
                fill='tonexty',
                fillcolor=colors[i].format(0.2),
                showlegend=False
            )

            rank_line_mean = go.Scatter(
                x=list(range(N_LAYERS)),
                y=ranks[:, i],
                mode='lines',
                name=axs_titles[i],
                line=dict(color=colors[i].format(1.0), width=1),
                legend='legend2',
            )

            rank_line_median = go.Scatter(
                x=list(range(N_LAYERS)),
                y=ranks_median[:, i],
                mode='lines',
                line=dict(width=1, dash='dot', color=colors[i].format(1.0)),
                name='Median',
                showlegend=False
            )

            fig.add_trace(rank_line_upper, row=next_row, col=1)
            fig.add_trace(rank_line_lower, row=next_row, col=1)
            fig.add_trace(rank_line_mean, row=next_row, col=1)
            fig.add_trace(rank_line_median, row=next_row, col=1)

    # Adjust layout for the inverted y-axis and other settings
    fig.update_yaxes(
        tickvals=axs_vals_flow,
        ticktext=[
            f'<span style="color:{colors[i].format(1.0)};">' + axs_titles[i] + '</span>'
            for i in range(num_rows)
        ],
        range=[0.2, 0.8],
        showline=True,
        linewidth=1,
        linecolor='black',
        ticks="outside",
        ticklen=6,
        row=1, col=1,
        tickangle=-90,  # Optional: rotate the labels
        title="Patching Flow"
    )

    if add_prob:
        fig.update_yaxes(range=[0, 1],row=2, col=1, showline=True, linewidth=1, linecolor='black', ticklen=3, tickwidth=0.1, ticks="outside", title="Answer Likelihood")
    if add_rank:
        fig.update_yaxes(range=[0, 500], title_text="Rank", row=3, col=1, showline=True, linewidth=1, linecolor='black', ticklen=3, tickwidth=0.1)

    fig.update_xaxes(
        tickmode='array',
        tickvals=list(range(N_LAYERS)),
        ticktext=[str(i) for i in range(N_LAYERS)],
        showgrid=True,
        ticks="outside",
        ticklen=6,
        minor_ticks="outside",
        tickwidth=1,
        showticklabels=True,
        range=[0, N_LAYERS - 0.5],
        title_text="Layer Number", 
    )    
    fig.update_xaxes(
        title_text=None,  # Remove the x-axis label for subplot 1
        row=1, col=1
    )
    
    # Adjust layout dimensions and margins
    height = 330 + (num_sp - 2) * 100
    fig.update_layout(height=height, width=800, title_text=title, margin=dict(l=40, r=40, t=100 if title else 40, b=0), plot_bgcolor='rgba(0,0,0,0)')

    # Legends
    fig.update_layout(
        legend1=dict(
            x=0.02,  
            y=1.0,  # Position the legend above the plot
            orientation="v",  # Horizontal orientation
            xanchor="left",  # Align the legend center with the x position
            yanchor="top",  # Align the bottom of the legend with the y position,
            # border
            bordercolor="black",  # Color of the border
            borderwidth=1,  # Width of the border
            bgcolor="white"  # Background color of the legend
        ),
        legend2=dict(
            x=0.02,  
            y=0.5,  # Position the legend above the plot
            orientation="v",  # Horizontal orientation
            xanchor="left",  # Align the legend center with the x position
            yanchor="top",  # Align the bottom of the legend with the y position,
            # border
            bordercolor="black",  # Color of the border
            borderwidth=1,  # Width of the border
            bgcolor="white"  # Background color of the legend
        ),
    )
    return fig