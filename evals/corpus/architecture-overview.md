> Fictional document for the DocPilot eval corpus. Lumen Labs, Lumen Search, and every person named here are invented.

# Lumen Search Architecture Overview

Owner: Tomas Okafor. Last updated: November 18, 2024.

This page is the starting point for understanding how Lumen Search works. It describes the major services, how a request flows through them, where data lives, and how the system is deployed. Team ownership of each service is listed in the Team Directory.

## Summary

Lumen Search is a hosted document-search API. Customers push documents through the ingest API and query them through the search API. Production runs on Amazon EKS (Kubernetes 1.29) in two AWS regions: us-west-2 is the primary region and eu-central-1 is the second region. Customers who sign up for EU hosting have their data pinned to eu-central-1, and their traffic never leaves that region.

## Services

- Beacon is the public API gateway. It terminates TLS on port 8443, enforces per-customer rate limits, and routes requests to internal services. Beacon applies a 2-second upstream timeout to every internal call.
- Keystone is the authentication service, listening on port 9100. It validates customer API keys and issues short-lived JWT access tokens that expire after 15 minutes.
- Prism is the query service, listening on port 7070. It parses a search request, fans it out to the relevant index shards, merges the results, and applies ranking.
- Riverbed is the ingestion pipeline. Its workers expose an admin and metrics port on 8081.
- Lattice is the index cluster, built on OpenSearch 2.11. Each region runs 48 primary shards with one replica per shard.

## Query path

A search request arrives at Beacon, which asks Keystone to validate the caller's token. Beacon caches successful validations for 60 seconds so that most requests skip the call to Keystone. The request then goes to Prism, which checks the Redis result cache, queries Lattice on a cache miss, and returns ranked results to the customer through Beacon.

## Ingest path

Customers upload documents with a POST to the documents endpoint. Beacon writes each raw document event to the Kafka topic ingest.raw.v3, hosted on Amazon MSK with 6 brokers per region. Riverbed workers consume that topic, extract text with Apache Tika, split the text into passages of at most 512 tokens, compute embeddings, and write the passages into Lattice. Riverbed autoscales between 12 and 80 worker pods based on consumer lag.

The freshness target is that 95% of new documents become searchable within 60 seconds of upload. Original files are also stored unchanged in a per-region S3 bucket so that indexes can be rebuilt from scratch.

## Data stores

- meta-db is a PostgreSQL 15 database on Amazon RDS. It stores accounts, API key metadata, index configuration, and the signing keys Keystone uses for JWTs.
- A Redis 7 cluster caches query results with a time-to-live of 300 seconds.
- S3 holds original documents and nightly exports.
- Lattice holds the searchable passages and their vectors.

## Observability

Metrics are collected by Prometheus and displayed in Grafana. Logs go to Loki, errors go to Sentry, and alerts page through PagerDuty. Distributed traces are exported with OpenTelemetry to Grafana Tempo, with 10% of requests sampled.

## Deployment

Every service is packaged as a container image built by GitHub Actions. Argo CD watches the deploy repository and syncs each cluster to the desired state, and Argo Rollouts handles canary traffic shifting. The rules for when and how changes reach production live in the Release Process document.

## Capacity notes

At the November 2024 peak, the platform served about 2,400 queries per second across both regions and ingested about 9 million documents per day. The us-west-2 region carries roughly 70% of query traffic. Lattice storage is expected to double by the end of 2025, mostly because Lumen Search v2 stores a vector for every passage.

## Known limitations

- There is no cross-region failover for EU-pinned customers, by design, because their data may not leave eu-central-1.
- meta-db is a single writer per region, which makes it a shared dependency for Keystone, Prism configuration lookups, and the admin console.
- Riverbed processes each document in a single worker, so very large scanned PDFs can take a long time to extract.
