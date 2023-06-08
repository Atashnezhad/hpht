from functools import wraps
import logging
# show the loger with info level
logging.basicConfig(level=logging.INFO)


def handle_default_values(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        il_names = kwargs.get("il_names")
        if not il_names:
            logging.info("No IL names provided, using default values")
            kwargs["il_names"] = ['HPyBF4_CP', 'HPyBr_CP']
        return func(*args, **kwargs)

    return wrapper