# third_party

Vendored build-time dependencies so a clean clone builds with no sibling
repositories or tokens.

## hailo_ipc_sdk-0.3.0-py3-none-any.whl

- Source: `ne503-aipc-sdks/python` (sibling SDK repo), copied 2026-08-26
- Version pinned to **0.3.0** to match the SDK shipped in the deployed
  `aipc/shelf-ops:0.3.0` image (verified via `pip show hailo-ipc-sdk`)
- Consumed by `scripts/build_showcase_artifacts.sh` (`find_default_wheel()`:
  `--wheel` > `third_party/` > `dist/` > sibling SDK repos)
- To upgrade: build the new wheel in the SDK repo, drop it here (keep only one
  version), rebuild showcases, then update the version note above
