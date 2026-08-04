# Contributing

Thank you for improving the NeoRuntime sample applications.

## What to Contribute

- New sample applications built with the public NeoRuntime SDK
- Bug fixes for existing app examples
- Documentation and manifest improvements
- Tests for reusable app logic

## App Guidelines

- Keep each application self-contained in its own directory.
- Place templates under `templates/`, focused examples under `examples/`,
  developer utilities under `tools/`, and full reference apps under
  `showcases/`.
- Do not commit generated packages, container image tarballs, model binaries,
  private datasets, device logs, or credentials.
- Use placeholder values in manifests and examples.
- Prefer environment variables for runtime configuration and secrets.
- Keep dependencies explicit in each app's `requirements.txt`, `Dockerfile`, or
  build documentation.

## Before Opening a Pull Request

```bash
python -m pytest
```

Run any app-specific tests or linters listed in that application's README.

## Security

Please do not file public issues for suspected vulnerabilities. Follow
[SECURITY.md](./SECURITY.md) instead.

## License

By contributing, you agree that your contribution will be licensed under the
MIT License used by this repository.
