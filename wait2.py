# The user issue says:
# "File: src/ferminet/train.py:196"
# "Current Code:"
# ```python
#        step_result = cast(tuple[Any, Any, Any, Any, Any], step_result)
#        new_params, new_opt_state, data, key, stats = step_result
#
#        energy_val = _to_host_scalar(stats.energy)
# ```
# BUT I don't see `_to_host_scalar` or `stats.energy` ANYWHERE in my codebase right now!
# `stats` is an array: `stats_host[ENERGY]` is used instead of `stats.energy`.
# Wait, let me check the git history.
