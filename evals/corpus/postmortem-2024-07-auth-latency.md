> Fictional document for the DocPilot eval corpus. Lumen Labs, Lumen Search, and every person named here are invented.

# Postmortem: Auth Latency Degradation on July 23, 2024

Status: Final. Severity: SEV2. Incident commander: Mei Lin. Author: Kwame Asante. Postmortem review held on July 30, 2024.

This is a blameless postmortem written to capture what happened, why, and what we changed afterwards.

## Summary

On July 23, 2024, between 16:12 and 17:48 UTC, a period of 96 minutes, token validation in the Keystone authentication service slowed dramatically. Keystone p99 validation latency rose from its normal 40 ms to 1.9 seconds. Because every search and ingest request is authenticated, the whole API slowed down, and about 6% of API requests failed with a timeout at the Beacon gateway. There was no data loss and no security exposure.

## Timeline (all times UTC)

- 15:35: Keystone 2.4.1 began its canary at 5% of traffic.
- 16:05: The canary passed its automated checks and Keystone 2.4.1 was promoted to all traffic.
- 16:12: Latency started climbing as cached signing keys expired across Keystone pods.
- 16:31: The first customer ticket arrived from an Enterprise customer reporting slow searches.
- 16:44: Customer support escalated to the on-call engineer and a SEV2 was declared.
- 17:20: The team identified the signing-key cache configuration as the root cause.
- 17:26: Keystone was rolled back to version 2.4.0.
- 17:48: Latency returned to normal across both regions.

## Root cause

Keystone 2.4.1 renamed the configuration key that sets the time-to-live of the JWKS signing-key cache. Our deployment configuration still used the old key name, which the new version silently ignored, so the cache TTL fell back to a default of zero. With a zero TTL, every token validation read the signing keys directly from meta-db instead of from memory. Under normal afternoon traffic, meta-db CPU climbed to 97%, and every query that touched meta-db slowed down, including Keystone's own reads.

The canary did not catch the problem because 5% of traffic was not enough load to saturate meta-db.

## Detection gap

At the time, the only Keystone alert was KeystoneErrorRate. Requests were slow rather than failing, so that alert never fired. The first signal was a customer ticket, which arrived 19 minutes after latency started rising, and the incident was declared 32 minutes after it started. Customers noticed the problem before our monitoring did.

## Resolution

Rolling back to Keystone 2.4.0 restored the old configuration key and the cache immediately. On July 25, 2024, the Identity team shipped Keystone 2.4.2, which sets the signing-key cache TTL to 10 minutes and refuses to start if the cache TTL configuration is missing or zero.

## Customer impact

No service credits were issued, because monthly query uptime stayed within every customer's contractual commitment. Beatriz Santos's team sent a written incident summary to every Enterprise customer that had opened a ticket.

## Action items

1. Add a KeystoneLatencyP99 alert. Owner: Mei Lin. Completed July 26, 2024.
2. Require a load test for services on the authentication path before they are promoted to production. Owner: Ravi Chandrasekar. Completed August 2024.
3. Require a Security team review for every Keystone change, including configuration-only changes. Owner: Fatima Al-Sayed. Completed August 9, 2024.
4. Add a meta-db read replica dedicated to Keystone signing-key reads. Owner: Marcus Lindqvist. Target: end of Q4 2024.
5. Add a startup check that logs every unrecognized configuration key as an error. Owner: Kwame Asante. Completed August 2, 2024.

## Lessons

Canary checks only catch problems that show up at canary scale. Any dependency that is shared across all pods, such as meta-db, can hide a problem until traffic is fully shifted. Load testing at realistic volume is now part of the release process for authentication services for this reason.
