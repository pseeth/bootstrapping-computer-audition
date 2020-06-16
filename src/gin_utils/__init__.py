from . import nussl_configurables
from gin.torch import external_configurables
import gin
import nussl
import torch
import inspect

def get_gin_macros():
    macros = {}
    for (scope, selector), config in gin.config._CONFIG.items():
        if selector == 'gin.macro':
            macros[scope] = config['value']
    return macros

def get_original_class(cls_name):
    cls_name = cls_name.split('.')
    # get the class from the top-level 
    # module in globals
    kls = globals()[cls_name[0]]
    # now find the original class
    for c in cls_name[1:]:
        kls = getattr(kls, c)
    return kls

def get_gin_argument(full_arg):
    try:
        arg_setting = gin.query_parameter(full_arg)
    except:
        return None
    macros = get_gin_macros()
    if str(arg_setting).startswith('%'):
        arg_setting = macros[str(arg_setting)[1:]]
    return arg_setting

@gin.configurable
def unginify(*args, kls=None, kls_name=None, **kwargs):
    if kls_name is None:
        kls_name = f"{kls.__module__}.{kls.__name__}"
    og_cls = get_original_class(kls_name)
    
    args_to_cls = inspect.getfullargspec(og_cls).args
    args_to_cls.append('init')
    scope = gin.current_scope_str()
    for arg in args_to_cls:
        queries = [
            f"{scope}/{kls_name}.{arg}",
            f"{kls_name}.{arg}",
        ]
        for query in queries:
            val = get_gin_argument(query)
            if val is not None:
                if isinstance(val, gin.config.ConfigurableReference):
                    val = val.configurable.fn_or_cls()
                kwargs[arg] = val
                break
    
    instantiated_cls = og_cls(*args, **kwargs)
    return instantiated_cls

@gin.configurable
def unginify_compose(tfm):
    gin_com = tfm()
    transforms = []
    for t in gin_com.transforms:
        t = unginify(t.__class__)
        transforms.append(t)
    com = nussl.datasets.transforms.Compose(transforms)
    return com
