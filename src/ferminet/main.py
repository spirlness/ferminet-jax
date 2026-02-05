import sys

from absl import app, flags, logging
from ml_collections import config_flags

from ferminet import base_config

_CONFIG = config_flags.DEFINE_config_file(
    "config",
    None,
    "Path to config file.",
    lock_config=True,
)

FLAGS = flags.FLAGS


def main(argv):
    if len(argv) > 1:
        raise app.UsageError("Too many command-line arguments.")

    cfg = _CONFIG.value
    if cfg is None:
        logging.info("No config provided, using default config")
        cfg = base_config.default()

    logging.info("FermiNet training")
    logging.info("Config:\n%s", cfg)

    try:
        from ferminet import train

        if train is None:
            logging.error(
                "train module is None - likely a silent import failure in __init__.py"
            )
            logging.error("Try importing directly: python -c 'import ferminet.train'")
            sys.exit(1)

        train.train(cfg)
    except ImportError as e:
        logging.error("Could not import train module: %s", e)
        logging.error("Make sure all dependencies are installed")
        sys.exit(1)
    except AttributeError as e:
        logging.error("AttributeError when calling train: %s", e)
        logging.error(
            "This may indicate a missing or failed import. Try: python -c 'import ferminet.train'"
        )
        sys.exit(1)


if __name__ == "__main__":
    flags.mark_flag_as_required("config")
    app.run(main)
