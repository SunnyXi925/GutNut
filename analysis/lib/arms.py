"""Non-overlapping prior arms. The microbial arm may never contain host information."""
MICRO_TRAITS = {'taxa', 'function'}
HOST_TRAITS = {'host'}
COMBO_TRAITS = {'host_taxa', 'host_function'}


def arm_of(cfg):
    t = cfg['trait']
    if t == 'uniform':
        return 'population'
    if t in HOST_TRAITS:
        return 'host'
    if t in MICRO_TRAITS:
        return 'micro'
    if t in COMBO_TRAITS:
        return 'host_micro'
    raise ValueError(t)


def arm_configs(all_configs, arm, food='masked'):
    return {k: v for k, v in all_configs.items()
            if arm_of(v) == arm and (food is None or v['food'] == food)}


def select(table, cfgs, arm, food='masked', metric='rmse'):
    """Inner-fold winner inside one arm, with an assertion against arm drift."""
    keys = set(arm_configs(cfgs, arm, food))
    sub = table[table.config.isin(keys)]
    assert len(sub), f'no candidate in arm {arm}'
    best = sub.sort_values([metric, 'config']).iloc[0].to_dict()
    assert arm_of(cfgs[best['config']]) == arm, 'selected config left its arm'
    return best
