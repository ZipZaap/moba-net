import argparse
from typing import Any

class SpacedDefaultsFormatter(argparse.RawTextHelpFormatter):
    """
    Help formatter that keeps raw text, shows defaults,
    adds a blank line after each option, and fixes the layout.
    """
    
    def __init__(self, *args, **kwargs):
        kwargs.setdefault('indent_increment', 2)
        kwargs.setdefault('max_help_position', 10)
        kwargs.setdefault('width', 120)
        super().__init__(*args, **kwargs)

    def _format_action(self, action):
        return super()._format_action(action) + '\n'

    def _format_action_invocation(self, action):
        shown_default = getattr(action, "shown_default", None)
        if (action.nargs == 0 or action.metavar == '') and shown_default is not None:
            return ', '.join(action.option_strings) + f' [default: {shown_default}]'
        return super()._format_action_invocation(action)

def parse_cli_args(cfg: dict[str, dict], 
                   *, 
                   inference: bool = False
                   ) -> dict[str, dict]:
    """
    Update the configuration dictionary with command line arguments.

    Parameters
    ----------
    cfg : dict[str, dict]
        Configuration dictionary with keys corresponding to parameter names.

    inference : bool
        If True, only inference-related keys will be considered.

    Returns
    -------
    cfg : dict[str, dict]
        Configuration dictionary updated with command line arguments.
    """

    inference_keys: list[str] = [
        "DATASET_DIR", 
        "NUM_WORKERS", 
        "CHECKPOINT", 
        "BATCH_SIZE", 
        "GPUs", 
        "CLS_THRESHOLD"
    ]

    type_mapping: dict[str, type] = {
        'int': int,
        'float': float,
        'str': str,
        'bool': bool,
        'list': list,
        'dict': dict
    }

    # If inference, filter out training-specific keys
    if inference:
        cfg = {k: v for k, v in cfg.items() if k in inference_keys}
        print(f"[INFO] Type `predict.py --help` for more information.")
    else:
        print(f"[INFO] Type `train.py --help` for more information.")

    # Create a parser with the SpacedDefaultsFormatter
    parser = argparse.ArgumentParser(
        description="",
        formatter_class=SpacedDefaultsFormatter
    )

    # Add arguments to the parser based on the configuration dictionary
    for name, param in cfg.items():
        kwargs: dict[str, Any] = {}
        shown_default: Any | None = None

        for key, value in param.items():
            if key == 'type':
                value = type_mapping[value]
                if value == bool:
                    kwargs['action'] = argparse.BooleanOptionalAction
                else:
                    kwargs[key] = value
            elif key == 'default':
                shown_default = value
            else:
                kwargs[key] = value
            kwargs['metavar'] = ''
            kwargs['default'] = argparse.SUPPRESS

        action = parser.add_argument(f'--{name.lower()}', **kwargs)
        if shown_default is not None:
            setattr(action, "shown_default", shown_default)

    # Parse the command line arguments and update the configuration
    args = vars(parser.parse_args())
    for key, value in args.items():
        if value is not None:
            cfg[key.upper()]['default'] = value

    return cfg