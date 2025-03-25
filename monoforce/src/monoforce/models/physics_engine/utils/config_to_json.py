import json
from functools import partial


def custom_encoder(obj):
    """
    Handles encoding of special types like classes and partial functions.
    """
    if isinstance(obj, type):  # For classes
        return {"__class__": obj.__module__ + "." + obj.__name__}  # Store class and module name
    if isinstance(obj, partial):  # For partial functions
        func = obj.func
        args = obj.args
        keywords = obj.keywords if obj.keywords else {}

        return {"__partial__": True, "func": func.__module__ + "." + func.__name__, "args": args, "keywords": keywords}

    return str(obj)  # Fallback: convert to string


def custom_decoder(obj):
    """
    Handles decoding of special types (classes and partial functions).
    """
    if "__class__" in obj:
        module_name, class_name = obj["__class__"].rsplit(".", 1)
        module = __import__(module_name, fromlist=[class_name])
        cls = getattr(module, class_name)
        return cls

    if "__partial__" in obj:
        module_name, func_name = obj["func"].rsplit(".", 1)
        module = __import__(module_name, fromlist=[func_name])
        func = getattr(module, func_name)
        return partial(func, *obj["args"], **obj["keywords"])
    return obj


def config_to_json(config):
    """
    Serializes the config to a JSON string.
    """
    return json.dumps(config, indent=4, default=custom_encoder)


# Serialization


def save_config(config, filename):
    with open(filename, "w") as f:
        json.dump(config, f, indent=4, default=custom_encoder)


# Deserialization


def load_config(filename):
    with open(filename, "r") as f:
        config = json.load(f, object_hook=custom_decoder)
    return config
