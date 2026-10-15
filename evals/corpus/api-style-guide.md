> Fictional document for the DocPilot eval corpus. Lumen Labs, Lumen Search, and every person named here are invented.

# Lumen Search API Style Guide

Owner: API Platform team. Version 3.1, published December 2, 2024.

This guide defines how every public Lumen Search endpoint should look and behave. Consistency matters more than any individual preference: a customer who learns one endpoint should be able to guess how the others work. Internal service-to-service APIs are encouraged, but not required, to follow the same rules.

## URLs and naming

- URL paths are lowercase, with words separated by hyphens, and collection names are plural nouns.
- A nested resource looks like /v2/search-indexes/{index_id}/documents/{document_id}.
- JSON field names always use snake_case, never camelCase.
- Every resource ID carries a short type prefix followed by an underscore: doc_ for documents, idx_ for indexes, and key_ for API keys.
- Actions that do not fit CRUD use a colon suffix on the resource, for example a reindex action on an index.

## Versioning

The major version appears in the URL path, and the current public version is v2. A breaking change is anything that removes or renames a field, changes a field's type, adds a new required parameter, or changes the meaning of an existing value. Breaking changes always require a new major version.

Additive changes, such as new optional fields or new endpoints, can ship within the current version without notice. Deprecated API versions remain fully supported for 12 months after the deprecation notice is published. During that window, every response from a deprecated version includes a Sunset header with the shutdown date.

## Pagination

List endpoints use cursor-based pagination only. The client passes page_token, and the response returns next_page_token, which is empty on the last page. The default page size is 50 items and the maximum page size is 500 items. Offset-based pagination is not allowed because it becomes slow and inconsistent on large indexes.

## Errors

Errors use a single envelope with three fields: code, message, and request_id. Error codes are written in UPPER_SNAKE_CASE, for example INDEX_NOT_FOUND or DOCUMENT_TOO_LARGE. Messages are written for humans and may change; codes are part of the contract and may not.

Status codes are used consistently:

- 400 for validation errors, with a details list naming each invalid field.
- 401 when credentials are missing or invalid, and 403 when valid credentials lack permission.
- 404 when the resource does not exist, and 409 for conflicts such as a duplicate document ID.
- 429 when a rate limit is exceeded, always with a Retry-After header given in seconds.

## Idempotency

All POST endpoints accept an Idempotency-Key header. Lumen Search stores the key and the original response for 24 hours, so a client that retries with the same key gets the same result instead of creating a duplicate.

## Timestamps and rate-limit headers

Timestamps are RFC 3339 strings in UTC with a Z suffix. Timestamp field names end in _at, for example created_at and updated_at. Every response includes the X-RateLimit-Limit, X-RateLimit-Remaining, and X-RateLimit-Reset headers so clients can slow down before they hit a 429.

## Review process

Every change to the public API must be approved by the API Review Board before it is implemented. The board meets every Thursday at 11:00 Pacific, and proposals must be submitted by end of day Tuesday to be discussed that week. A proposal needs approval from at least two board members, one of whom must be from the API Platform team. Urgent security fixes can skip the board but must be reviewed at the next meeting.

## Linting

OpenAPI specs are linted in CI with Spectral using the shared ruleset named lumen-api-rules. A failed lint blocks the merge. If you believe a rule is wrong, propose a change to the ruleset at the API Review Board rather than disabling it in your spec.
