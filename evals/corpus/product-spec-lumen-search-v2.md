> Fictional document for the DocPilot eval corpus. Lumen Labs, Lumen Search, and every person named here are invented.

# Product Spec: Lumen Search v2

Product manager: Lena Kowalski. Engineering lead: Aiko Tanaka. Status: Approved. Spec version 1.4, October 21, 2024.

## Background

Lumen Search v1 launched in 2022 and supports keyword search only, ranked with BM25. Over the past year, the most common request from customers has been semantic search that understands meaning, not just matching words. Several prospects told the sales team they chose a competitor because v1 could not find documents that used different wording from the query. Lumen Search v2 adds semantic retrieval while keeping the simple API that customers like.

## Goals

- Hybrid retrieval that combines BM25 keyword scores with dense vector similarity, merged using reciprocal rank fusion.
- Reranking of the top 100 candidates with Lumen Rerank-S, a small cross-encoder model trained in house.
- Raise the maximum document size from 5 MB in v1 to 20 MB in v2.
- Support 42 languages for both text extraction and embeddings.
- Generate 768-dimensional embeddings with the in-house model lumen-embed-2.

## Non-goals

- On-premises deployment is out of scope for v2.
- Image and video search are out of scope.
- Fine-tuning custom models for individual customers is out of scope, although it may come in a later version.

## Success metrics

- Relevance: NDCG@10 at least 12% higher than v1 on the LumenBench-2 benchmark, which contains 4,200 labeled queries.
- Latency: p95 query latency under 300 ms for indexes of up to 10 million documents, including reranking.
- Availability: an internal availability SLO of 99.95% per calendar month for the query API.
- Adoption: 40 design-partner customers using v2 during the beta.

## Pricing and limits

- Starter costs $49 per month and includes up to 100,000 documents and 10 queries per second. Reranking is not included in Starter.
- Growth costs $499 per month and includes up to 5 million documents and 100 queries per second.
- Enterprise uses custom pricing, includes dedicated capacity, and comes with the Enterprise support tier described below.

Customers who exceed their document limit can keep querying, but new uploads are rejected with the error code PLAN_LIMIT_EXCEEDED until they upgrade or delete documents.

## API changes

All v2 endpoints live under the /v2/ path prefix and follow the API Style Guide. The search request gains a mode parameter with the values keyword, semantic, and hybrid, and hybrid is the default. Each search result gains a score_breakdown object that shows the keyword score, vector score, and rerank score separately, so customers can debug ranking.

## Timeline

- Private beta: November 12, 2024, with the design-partner customers.
- Public beta: January 14, 2025.
- General availability: March 4, 2025.

The v1 API deprecation notice will be published on the day v2 reaches general availability. Following the standard deprecation window in the API Style Guide, the v1 API will be shut down on March 4, 2026.

## Customer SLA and support at GA

The commitments in this section take effect on March 4, 2025, together with general availability, and are owned by the Customer Support team. If they ever disagree with a signed customer contract, the contract takes precedence.

### Uptime commitment

Lumen Labs commits to 99.9% monthly uptime for the query API on the Growth and Enterprise plans. The Starter plan has no uptime commitment. Uptime is measured by external probes that run a test query from three locations every minute; a minute counts as down if at least two of the three probes fail.

### Service credits

When monthly uptime falls below the commitment, customers receive a service credit applied to their next invoice:

- Below 99.9%: a credit of 10% of the monthly fee.
- Below 99.0%: a credit of 25% of the monthly fee.
- Below 95.0%: a credit of 50% of the monthly fee.

Customers must request a credit within 30 days after the end of the affected month. Scheduled maintenance does not count as downtime if it is announced at least 72 hours in advance and totals no more than 4 hours in a month.

### Support tiers

- Starter: email support only, with a first response within 2 business days.
- Growth: email and in-app chat, with a first response within 4 business hours.
- Enterprise: 24/7 phone support and a dedicated Slack Connect channel, a first response within 30 minutes for P1 tickets, and a named technical account manager.

### Escalation to engineering

Tickets are prioritized from P1, meaning production is down, to P4, meaning a feature request. For any P1 ticket, the support engineer pages the primary on-call engineer through PagerDuty and asks for an incident of at least SEV2 to be declared. P2 tickets that remain unresolved after 4 hours are escalated to the owning engineering team's lead.

The Lisbon support team covers 08:00 to 20:00 Lisbon time on weekdays for Growth customers. Enterprise customers are covered around the clock, with nights and weekends handled by an on-call support engineer who rotates weekly.

## Risks

- Reranking cost: GPU inference for Lumen Rerank-S is estimated at $38,000 per month at expected GA traffic. Mitigation: cache rerank results alongside query results.
- Latency: reranking adds roughly 60 ms at p95. Mitigation: limit reranking to the top 100 candidates and skip it entirely when the customer requests keyword mode.
- Storage: vectors roughly double the size of each index, which the Infrastructure team has included in its 2025 capacity plan.

## Open questions

- Should Starter customers be able to buy reranking as a paid add-on? Lena Kowalski will decide after the public beta.
- Should v2 offer a filter-only mode for customers who use Lumen Search as a structured database? No decision yet.
