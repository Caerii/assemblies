# Contributing

Contributions to the **neural-assemblies** project are welcome.

## How to contribute

- **Bug reports and feature ideas**: Open an [issue](https://github.com/Caerii/assemblies/issues) on GitHub.
- **Code or docs changes**: Open a pull request against the default branch. Ensure tests pass (`uv run pytest neural_assemblies/tests/ -q`) and, if you touch packaging, see [packaging.md](packaging.md).

## Development setup

From the repo root:

```bash
uv sync
uv run pytest neural_assemblies/tests/ -q
uv run ruff check neural_assemblies/
```

See the root [README.md](../README.md) for install options and [packaging.md](packaging.md) for maintainer release steps.

## License

By contributing, you agree that your contributions will be licensed under the same [MIT License](../LICENSE) as the project.
