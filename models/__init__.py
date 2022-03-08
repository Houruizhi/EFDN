from importlib import import_module
def make_model(args):
    module = import_module('models.' + args.model_name.lower())
    model = module.make_model(args)
    return model