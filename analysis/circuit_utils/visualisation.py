import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from plotly.colors import n_colors
import torch
import numpy as np
import einops
from pycolors import TailwindColorPalette, to_rgb

COLORS = TailwindColorPalette()


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


def get_name(head, shorten=False):
    """Returns a formatted name based on the head type."""
    if head.startswith('q'):
        return f'Attn Query{"<br>" if shorten else " "}(all heads)'
    elif head.startswith('o'):
        return "MHA output" #f"$\\text{{MHA out }} a $"
    elif head.startswith('m'):
        return 'MLP'
    else:
        return head

def get_flow_chart_params(head, q_y_offset, q_x_offset):
    """Returns the symbol, x_offset, and y_offset based on the head type."""
    if head.startswith('q'):
        return 'circle', -RECT_WIDTH, -2 * q_y_offset, head
    elif head.startswith('o'):
        return 'square', -0.5 * RECT_WIDTH, 2 * q_y_offset, "a"
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
        marker=dict(size=13, color=color, symbol=symbol, line=dict(color="black", width=1) if head.endswith('cross') else None),
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
        subplot_titles=["<b>Patching Flow</b>"] + (["<b>Answer Likelihood</b>"] if add_prob else []) + (["<b>Answer Rank</b>"] if add_rank else []),
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
        return 'circle', -RECT_WIDTH, -2 * q_y_offset, head
    elif head.startswith('o'):
        return 'square', -0.5 * RECT_WIDTH, 2 * q_y_offset, ""
    elif head.startswith('mlp'):
        head = head.replace('mlp', 'm')
        return 'square', 2 * q_x_offset, 3.5 * q_y_offset, head
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
        mode='markers',
        text=[head.split("cross")[0]] * len(x_vals),
        marker=dict(size=18, color=color, symbol=symbol, line=dict(color="black", width=1) if head.endswith('cross') else None),
        showlegend=legend,
        textposition='middle center',
        textfont=dict(
            color="black",
        ),
        name=get_name(head, shorten=False)
    ), row=1, col=1)
    
    for xv in x_vals:
        #add annotation
        fig.add_annotation(
            x=xv,
            y=y_pos+0.01,
            xref="x1", yref="y1",
            text=head.split("cross")[0],
            showarrow=False,
            font=dict(size=14, color="white"),

        )


def add_flow_chart_l(fig, y_pos, layers, color, residual_color, q_y_offset=0, q_x_offset=0, head='q', shorten_legend=False):
    """Adds a flow chart to the figure."""
    symbol, x_offset, y_offset, head = get_flow_chart_params_l(head, q_y_offset, q_x_offset)

    if not head.endswith('cross'):
        add_marker_trace_l(fig, [el for el in layers], y_pos + x_offset/2 + 0.01, head, color, symbol)
    
    for x in layers:
        add_rectangle_l(fig, head, x, q_y_offset, residual_color)
        
    add_marker_trace_l(fig, [el for el in layers], MAIN_LINE + x_offset, head, color, symbol)

    if head.endswith('cross'):
        fig.add_trace(go.Scatter(
            x=[el + y_offset for el in layers],
            y=[MAIN_LINE + x_offset ] * len(layers),
            mode='markers',
            marker=dict(size=12, color="red", symbol="line-ne", line=dict(color="red", width=1)),
            showlegend=False,
            name="Avg Patch"
        ), row=1, col=1)

    for x in layers:
        if not head.endswith('cross'):
            fig.add_annotation(
                ax=x , ay=y_pos + x_offset/2 - 0.04,
                axref="x1", ayref="y1",
                x=x, y=MAIN_LINE + x_offset - 0.1,
                xref="x1", yref="y1",
                showarrow=True,
                arrowhead=2,
                arrowsize=1,
                arrowwidth=1.5,
                arrowcolor=color
            )
            
    

def get_colors(a_name, b_name):
    a_name = a_name.lower()
    b_name = b_name.lower()
    if "ctx" in a_name and "ctx" in b_name:
        return COLORS.get_shade(6, 700), COLORS.get_shade(6, 500)
    elif "pri" in a_name and "pri" in b_name:
        return COLORS.get_shade(3, 400), COLORS.get_shade(2, 500)
    elif "pri" in a_name and "ctx" in b_name:
        return COLORS.get_shade(3, 400), COLORS.get_shade(6, 700)
    elif "ctx" in a_name and "pri" in b_name:
        return COLORS.get_shade(6, 700), COLORS.get_shade(3, 400),
    

def create_patch_scope_lplot(probs, ranks, a_layers, b_layers, avg_layers, aggregation="median", a_title="Alt CTX", b_title="CTX", c_title="PRIOR", title=None, q_x_offset=0.03, q_y_offset=0.1, N_LAYERS=32, add_rank=True, add_prob=True):
    """Creates a patch scope plot with flow charts and line plots."""
    probs, std_devs_probs, probs_median = probs
    ranks, std_devs_ranks, ranks_median = ranks

    num_sp = 3 if add_rank and add_prob else 2
    num_rows = probs.shape[1]
    # colors = ['rgba(31, 119, 180, {})', 'rgba(255, 127, 14, {})','rgba(44, 160, 44, {})']

    colors = [f"rgba({','.join(str(el) for el in to_rgb(c))}, {{}})" for c in get_colors(a_title, c_title)]

    fig = sp.make_subplots(
        rows=num_sp, cols=1,  # Adjusted to have rows instead of columns
        shared_xaxes=True, vertical_spacing=0.40,
        specs=[[{'type': 'scatter'}]] + [[{'type': 'scatter'}] for _ in range(num_sp-1)],
        row_heights=[0.4, 0.6],
        subplot_titles=["<b>Patching Flow</b>"] + (["<b>Answer Probability</b>"] if add_prob else []) + (["Answer Rank"] if add_rank else []),
    )

    # Add flow chart  
    fig.add_trace(go.Scatter(
        x=[0, N_LAYERS - 1],
        y=[MAIN_LINE, MAIN_LINE],
        mode='lines',
        line=dict(color=colors[num_rows-1].format(1.0), width=2, dash='dot'),
        showlegend=True,
        legend="legend1",
        name="Residual" #"$\\text{Residual}$"
    ), row=1, col=1)

    for head in a_layers:
        add_flow_chart_l(fig, PATCH_ALT, a_layers[head], q_x_offset=q_x_offset, q_y_offset=q_y_offset, head=head, color=colors[0].format(1.0), residual_color=colors[-1].format(1.0))
    for head in b_layers:
        add_flow_chart_l(fig, PATCH_MAIN, b_layers[head], q_x_offset=q_x_offset, q_y_offset=q_y_offset, head=head, color=colors[1].format(1.0), residual_color=colors[-1].format(1.0))
    for head in avg_layers:
        add_flow_chart_l(fig, MAIN_LINE, avg_layers[head], q_x_offset=q_x_offset, q_y_offset=q_y_offset, head=head+'cross', color="white", residual_color=colors[1].format(1.0))


    axs_titles = [a_title, c_title] if num_rows == 2 else [a_title, b_title, c_title]
    axs_vals_flow = [PATCH_ALT, MAIN_LINE] if num_rows == 2 else [PATCH_ALT, PATCH_MAIN, MAIN_LINE]
    axs_vals_data = [0, 1] if num_rows == 2 else [0, 1, 2]
    next_row = 2
    
    # Add legend for flow chart symbols
    for head, symbol, color in zip(['m', 'q', 'o'], ["diamond", "circle", "square"], ["white"] * 3):
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
                name=axs_titles[i],#f"$\\text{{{axs_titles[i]}}}$",
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
            name="Mean",#"$\\text{Mean}$",
            legendgroup="meta"

        ), row=next_row, col=1)

        fig.add_trace(go.Scatter(
            x=[None] ,
            y=[None],
            mode='lines',
            showlegend=True,
            legend='legend2',
            line=dict(width=1, dash='dot', color = 'black'),
            name="Median",#"$\\text{Median}$",
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
                name=f"${axs_titles[i]}$",
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

        # legend entries for median and mean
        fig.add_trace(go.Scatter(
            x=[None] ,
            y=[None],
            mode='lines',
            showlegend=True,
            legend='legend2',
            line=dict(width=1,color = 'black'),
            name="Mean",#"$\\text{Mean}$",
            legendgroup="meta"

        ), row=next_row, col=1)

        fig.add_trace(go.Scatter(
            x=[None] ,
            y=[None],
            mode='lines',
            showlegend=True,
            legend='legend2',
            line=dict(width=1, dash='dot', color = 'black'),
            name="$\\text{Median}$",
            legendgroup="meta"

        ), row=next_row, col=1)
        

        # adjust y-axis range
        fig.update_yaxes(range=[0, 500], row=2, col=1)

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
        tickangle=-45,  # Optional: rotate the labels
    )

    if add_prob:
        fig.update_yaxes(range=[0, 1],row=2, col=1, showline=True, linewidth=1, linecolor='black', ticklen=3, tickwidth=0.1, ticks="outside")
    if add_rank:
        fig.update_yaxes(range=[0, 500], title_text="Rank", row=3, col=1, showline=True, linewidth=1, linecolor='black', ticklen=3, tickwidth=0.1)

    fig.update_xaxes(
        tickmode='array',
        tickvals=list(range(N_LAYERS)),
        ticktext=[str(i) for i in range(N_LAYERS)],
        showgrid=True,tickfont=dict(size=18),
        ticks="outside",
        ticklen=6,
        minor_ticks="outside",
        tickwidth=1,
        showticklabels=True,
        range=[0, N_LAYERS - 0.5],
    )    
    
    fig.update_xaxes(
        title=dict(text= "Layer Number", font=dict(size=18)), 
        row=2, col=1,
    )
    
    # Adjust layout dimensions and margins
    height = 360 + (num_sp - 2) * 100
    fig.update_layout(height=height, width=800, title_text=title, margin=dict(l=40, r=40, t=100 if title else 40, b=0), plot_bgcolor='rgba(0,0,0,0)')

  # increase font size
    fig.update_layout(font=dict(size=25))
    fig.update_annotations(font=dict(size=25))

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
            bgcolor="rgba(255,255,255,0.9)",
            font=dict(size=20)
            ),
        legend2=dict(
            x=0.02,  
            y=0.36,  # Position the legend above the plot
            orientation="h",  # Horizontal orientation
            xanchor="left",  # Align the legend center with the x position
            yanchor="top",  # Align the bottom of the legend with the y position,
            # border
            bordercolor="black",  # Color of the border
            borderwidth=1,  # Width of the border
            bgcolor="rgba(255,255,255,0.9)",
            font=dict(size=20)
        ),
    )

  
    return fig

def jupyter_enable_mathjax():
    import plotly
    from IPython.display import display, HTML

    plotly.offline.init_notebook_mode()
    display(HTML(
        '<script type="text/javascript" async src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.1/MathJax.js?config=TeX-MML-AM_SVG"></script>'
    ))


def get_label_color(label, COLORS):
    if 'FT' in label:
        return COLORS.get_shade(0, 700)
    elif 'FS' in label or 'ICL' in label:
        return COLORS.get_shade(9, 600)
    elif "ZS" in label:
        return COLORS.get_shade(6, 600)
    else:
        return 'black'  # Default color
        
def format_label(label, latex=False):

    if latex:
        label = label.replace("INSTRUCT ", "INSTR ")
        label = label.replace("FLOAT", ' \\numberemoji')
        label = label.replace("INSTRUCTION", ' \\pointemoji')
        label = label.replace("FS", "ICL")
        words = label.replace("-", " ").split()
        colored_label = ''.join(['\\textcolor{{{0}}}{{{{\\textbf{{{1}}}}}}}'.format(get_label_color(word, COLORS), word) if word in ["ICL", "FT", "ZS"] else '\\textcolor{{{0}}}{{{1}}}'.format(get_label_color(word, COLORS), word) for word in words])
    else:
        label = label.replace("INSTRUCT ", "INSTR ")
        label = label.replace("FLOAT", ' 1️⃣')
        label = label.replace("INSTRUCTION", ' 🫵')
        label = label.replace("FS", "ICL")
        words = label.replace("-", " ").split()
        colored_label = ''.join([f'<span style="color:{get_label_color(word, COLORS)};">{"<b>" if word in ["ICL", "FT", "ZS"] else ""} {word} {"</b>" if word in ["ICL", "FT", "ZS"] else ""}</span>' for word in words])
    return colored_label

def to_standart_label(label):
    label = label.replace("_", " ").replace("cwf", "").replace("instruction", "INSTRUCTION").replace("float", "FLOAT").replace("zs", "ZS").replace("fs", "FS").replace("ft", "FT")
    label = label.replace("base", "BASE").replace("instruct", "INSTRUCT")
    return label


column_map = {
    "baseline": "Baseline: Intent Instruction",
    "with_instruction": "Steering: Same Instruction",
    "against_instruction": "Steering: Opposite Instruction",
    "one_word": "Steering: Only One Word Instruction",
    "one_word_instruction": "Steering: One Word Instruction and Same Instruction",
    "no_instruction": "Steering: No Instruction",
    "baseline_one_word_instruction": "Baseline: Intent + One Word Instruction"
}
metric_map = {
    "acc": "Accuracy",
    "pair_acc": "PairAcc"
}


ORDER = ["instruct_ft_instruction", "instruct_ft_float", "base_ft_instruction", "base_ft_float", "instruct_fs_instruction", "instruct_fs_float", "base_fs_instruction", "base_fs_float", "instruct_zs_instruction", "instruct_zs_float", "base_zs_instruction", "base_zs_float"]
def plot_das_results(data, metric='accuracy', columns=None, use_one_word_baseline_for_zs=False, COLORS=TailwindColorPalette(), extended_legend=False):
    if use_one_word_baseline_for_zs:
        for key in data.keys():
            if "zs" in key:
                data[key]["baseline"] = data[key]["baseline_one_word_instruction"]
                data[key]["no_instruction"] = data[key]["one_word"]
    if columns is None:
        columns = ['baseline', 'with_instruction', 'against_instruction', 'one_word', 'one_word_instruction', 'no_instruction']
    
    fig = go.Figure()
    
    # order rows
    _data = {key: data[key] for key in ORDER if key in data}

    color_no_instruction = to_rgb(COLORS.get_shade(1, 500))
    color_one_word = to_rgb(COLORS.get_shade(1, 500))
    color_against = to_rgb(COLORS.get_shade(1, 500))
    color_baseline = to_rgb(COLORS.get_shade(4, 300))
    # Define colors for each column
    colors = {
        'baseline': [f"rgb({','.join(map(str, color_baseline))})"] * len(_data),
        'no_instruction': [f"rgb({','.join(map(str, color_no_instruction))})"] * len(_data),
        'against_instruction': [f"rgb({','.join(map(str, color_against))})"] * len(_data),
        'one_word': [f"rgb({','.join(map(str, color_one_word))})"] * len(_data),
    }
    for column in columns:
        y_values = []
        x_labels = []
        for key in _data.keys():
            if column in data[key].columns:
                value = float(data[key].loc[data[key]['Unnamed: 0'] == metric, column].values[0])
                y_values.append(value)
                x_labels.append(format_label(to_standart_label(key)))
        
        print(column, y_values)
        fig.add_trace(go.Bar(
            name=column_map[column],
            x=x_labels,
            y=y_values,
            text=[f'{v:.2f}' if v != 0 else '0' for v in y_values],
            textposition=['auto' if v != 0 else 'outside' for v in y_values],
            # textfont=dict(size=20),
            marker_color=colors[column]
        ))

    fig.update_layout(
        barmode='group',
        # title=f'Feature F_{{w}} Causality - {metric_map[metric]}',
        # xaxis_title='Model Configuration',
        yaxis_title=metric_map[metric],
        font=dict(size=16),
        xaxis=dict(tickangle=-45),
        width=2000,
        height=700
    )

    fig.update_layout(legend=dict(
        yanchor="bottom",
        y=0.88,
        xanchor="right",
        x=0.992,
        orientation="h",
        borderwidth=4,
        bordercolor="white"
    ))
    # xrange
    fig.update_yaxes(range=[0.0, 1.0])
    # font
    fig.update_layout(font=dict(size=25))
    
    
    if extended_legend:
        #Add custom legend for color codes
        custom_annotations = [
            go.Scatter(
                x=[None],
                y=[None],
                mode='markers',
                marker=dict(
                    size=10,
                    color=get_label_color("FT", COLORS)
                ),
                legendgroup='config',
                showlegend=True,
                name=' Finetuning (FT)',
                legend = 'legend2'
            ),
            go.Scatter(
                x=[None],
                y=[None],
                mode='markers',
                marker=dict(
                    size=10,
                    color=get_label_color("FS", COLORS)
                ),
                legendgroup='config',
                showlegend=True,
                name=' In-Context Learning (ICL)',
                legend = 'legend2'

            ),
            go.Scatter(
                x=[None],
                y=[None],
                mode='markers',
                marker=dict(
                    size=10,
                    color=get_label_color("ZS", COLORS)
                ),
                legendgroup='config',
                showlegend=True,
                name=' Zero-Shot (ZS)',
                legend = 'legend2'
            )
        ]

        # Add the custom legend to the figure
        for annotation in custom_annotations:
            fig.add_trace(annotation)


        # Add custom legend using annotations
        legend_annotations = [
            dict(
            x=.995,
            y=0.79,
            xref="paper",
            yref="paper",
            text=f"🫵  IF = instruction<br>1️⃣  IF = float",
            showarrow=False,
            font=dict(size=20),
            align="left",
            bgcolor="rgba(255, 255, 255, 0.9)",
            borderpad=10,
            xanchor="right",
            height=48,
            width=292,
            yanchor="top")
            
        ]
        fig.update_layout(annotations=legend_annotations)

    fig.update_layout(legend2=dict(
        yanchor="top",
        y=0.98,
        xanchor="right",
        x=0.995, #48
        orientation="v",
        bgcolor="rgba(255, 255, 255, 0.9)"
    ), margin=dict(l=0, r=0, t=0, b=0))

    return fig
