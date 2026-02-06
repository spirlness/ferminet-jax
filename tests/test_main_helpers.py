from ferminet import base_config, main


def test_apply_checkpoint_overrides_restore_path_precedence():
    cfg = base_config.default()
    updated = main._apply_checkpoint_overrides(
        cfg,
        restore_path="/tmp/checkpoint_00000010.pkl",
        resume_latest=True,
        save_path="/tmp/ckpt-dir",
    )
    assert updated.log.save_path == "/tmp/ckpt-dir"
    assert updated.log.restore_path == "/tmp/checkpoint_00000010.pkl"
    assert updated.log.restore is False


def test_apply_checkpoint_overrides_resume_latest():
    cfg = base_config.default()
    updated = main._apply_checkpoint_overrides(
        cfg,
        restore_path=None,
        resume_latest=True,
        save_path=None,
    )
    assert updated.log.restore is True
    assert updated.log.restore_path is None
