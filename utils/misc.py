import torch

def method_name(params):
    name = ''
    if params.ft_attn_module:
        name += 'attn_'
        name += params.ft_attn_module + '_'
        name += params.ft_attn_mode + '_'
        if params.ft_attn_mode == 'parallel':
            name += params.ft_attn_ln + '_'
        if params.ft_attn_module == 'adapter':
            name += str(params.adapter_bottleneck) + '_'
            name += params.adapter_init + '_'
            name += str(params.adapter_scaler) + '_'
        elif params.ft_attn_module == 'convpass':
            name += str(params.convpass_xavier_init) + '_'
            name += str(params.convpass_scaler) + '_'
        elif params.ft_mlp_module == 'repadapter':
            name += str(params.repadapter_scaler) + '_'
        else:
            raise NotImplementedError
    if params.ft_mlp_module:
        name += 'mlp_'
        name += params.ft_mlp_module + '_'
        name += params.ft_mlp_mode + '_'
        if params.ft_attn_mode == 'parallel':
            name += params.ft_attn_ln + '_'
        if params.ft_mlp_module == 'adapter':
            name += str(params.adapter_bottleneck) + '_'
            name += params.adapter_init + '_'
            name += str(params.adapter_scaler) + '_'
        elif params.ft_mlp_module == 'convpass':
            name += str(params.convpass_xavier_init) + '_'
            name += str(params.convpass_scaler) + '_'
        elif params.ft_mlp_module == 'repadapter':
            name += str(params.repadapter_scaler) + '_'
        else:
            raise NotImplementedError
    if params.vpt_mode:
        name += 'vpt_'
        name += params.vpt_mode + '_'
        name += str(params.vpt_num) + '_'
        name += str(params.vpt_layer) + '_'
    if params.ssf:
        name += 'ssf_'
    if params.lora_bottleneck > 0:
        name += 'lora_' + str(params.lora_bottleneck) + '_'
    if params.fact_type:
        name += 'fact_' + params.fact_type + '_' + str(params.fact_dim) + '_' + str(params.fact_scaler) + '_'
    if params.bitfit:
        name += 'bitfit_'
    if params.vqt_num > 0:
        name += 'vqt_' + str(params.vqt_num) + '_'
    if params.mlp_index:
        name += 'mlp_' + str(params.mlp_index) + '_' + params.mlp_type + '_'
    if params.attention_index:
        name += 'attn_' + str(params.attention_index) + '_' + params.attention_type + '_'
    if params.ln:
        name += 'ln_'
    if params.difffit:
        name += 'difffit_'
    if params.full:
        name += 'full_'
    if params.block_index:
        name += 'block_' + str(params.block_index) + '_'
    #####if nothing, linear
    if name == '':
        name += 'linear' + '_'
    name = name.rstrip('_')
    return name