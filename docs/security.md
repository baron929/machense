# Security

## Implemented

- Pydantic request validation and bounded numeric sensor values.
- Generic prediction error responses that do not return internal exception text.
- Non-root Docker runtime configuration.
- `.dockerignore` exclusions for local caches, models, and environment files.
- GitHub Actions `contents: read` permission.
- Dependency audit step configured in CI with `pip-audit`.
- MLflow skops serialization trusts only `sklearn.tree._tree.Tree`, the exact
	internal type required by the Random Forest baseline.

## Not yet implemented

Authentication, authorization, rate limiting, request-size limits, secret-manager integration, SBOM generation, image signing, and container scanning are not currently verified in this repository. Do not expose model-management functionality publicly until authorization exists.

## Secret handling

Do not commit credentials, tokens, private keys, production URLs, or personal data. Use environment variables or a deployment secret manager. Local `.env` files are ignored by the container build context.

## Residual risks

The service has no identity boundary and should be treated as an internal/local demo until placed behind an authenticated, rate-limited edge. Dependency and image scanning must run successfully in CI before describing a release as hardened.