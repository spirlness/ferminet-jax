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
_RESTORE_PATH = flags.DEFINE_string(
    "restore_path",
    None,
    "Optional checkpoint file path to resume training from.",
)
_RESUME_LATEST = flags.DEFINE_bool(
    "resume_latest",
    False,
    "Resume from latest checkpoint under cfg.log.save_path.",
)
_SAVE_PATH = flags.DEFINE_string(
    "save_path",
    None,
    "Optional override for cfg.log.save_path.",
)

FLAGS = flags.FLAGS


def _safe_flag_value(flag_holder, default):
    """Return flag value if parsed, otherwise default."""
    try:
        return flag_holder.value
    except flags.UnparsedFlagAccessError:
        return default


def _apply_checkpoint_overrides(
    cfg,
    restore_path: str | None,
    resume_latest: bool,
    save_path: str | None,
):
    """Apply CLI checkpoint options to config in-place."""
    with cfg.unlocked():
        if save_path:
            cfg.log.save_path = save_path
        if restore_path:
            cfg.log.restore_path = restore_path
            cfg.log.restore = False
        elif resume_latest:
            cfg.log.restore = True
            cfg.log.restore_path = None
    return cfg


def main(argv):
    if len(argv) > 1:
        raise app.UsageError("Too many command-line arguments.")

    cfg = _CONFIG.value
    if cfg is None:
        logging.info("No config provided, using default config")
        cfg = base_config.default()

    cfg = _apply_checkpoint_overrides(
        cfg,
        _safe_flag_value(_RESTORE_PATH, None),
        bool(_safe_flag_value(_RESUME_LATEST, False)),
        _safe_flag_value(_SAVE_PATH, None),
    )

    restore_path_value = _safe_flag_value(_RESTORE_PATH, None)
    resume_latest_value = bool(_safe_flag_value(_RESUME_LATEST, False))
    if restore_path_value and resume_latest_value:
        logging.warning(
            "Both --restore_path and --resume_latest were set; using --restore_path"
        )

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
