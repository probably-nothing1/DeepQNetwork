def polyak_averaging(target, signal, alpha=0.995):
    target_params = dict(target.named_parameters())

    for name, param in signal.named_parameters():
        if name in target_params:
            target_params[name].data *= alpha
            target_params[name].data += (1 - alpha) * param.data
