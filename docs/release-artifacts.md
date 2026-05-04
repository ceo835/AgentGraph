# Release Artifacts

The `v1.0.0` reproducibility bundle lives at:

```text
dist/
```

It is expected to contain:

- built wheel files for this package
- sample serialized `AgentState` payloads
- SQLite checkpoint fixtures
- a manifest describing the captured assets

Use this bundle for:

- smoke validation of the published package
- regression testing around checkpoint compatibility
- reproducing HITL and async memory sync behavior
