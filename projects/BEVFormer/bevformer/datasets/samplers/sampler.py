from mmengine.registry import Registry  # , build_from_cfg

SAMPLER = Registry("sampler")


def build_sampler(cfg, default_args):
    return SAMPLER.build(cfg, default_args)
