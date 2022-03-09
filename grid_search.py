from libraries_import_all import *

""" The configuration space of search we want to explore. """

def config_is_well_typed(config, full_search_options):
    return set(config.keys()).issubset(set(full_search_options.keys())) and \
           all(v in full_search_options[k] for k,v in config.items())

def iterate_configurations(gs_all_configs):
    config_list = list(gs_all_configs.items())
    config_choices = []
    def back_tracking(i):
        if i==len(config_list):
            yield { config_list[i][0]: config_list[i][1][idx] for i,idx in enumerate(config_choices) }
        else:
            for idx in range(len(config_list[i][1])):
                config_choices.append(idx)
                yield from back_tracking(i+1)
                config_choices.pop()
    yield from back_tracking(0)