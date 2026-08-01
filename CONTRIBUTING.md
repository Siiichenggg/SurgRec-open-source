# Contributing

Thank you for considering a contribution. Please open an issue before large changes and include enough information to reproduce bugs.

Before submitting a pull request:

- Format scripts consistently with the existing style.
- Do not commit raw videos, checkpoints, generated outputs, logs, or private paths.
- Run `python -m pytest tests -q` and the dry-run in `INSTALL.md`, or explain why they could not be run.

CI runs the same checks on every pull request: byte-compilation, the test suite,
shell syntax, config and citation metadata, that paths named in the docs exist,
that no machine-specific path or credential is present, and that no large file
is tracked.
