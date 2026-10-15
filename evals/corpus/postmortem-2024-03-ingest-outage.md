> Fictional document for the DocPilot eval corpus. Lumen Labs, Lumen Search, and every person named here are invented.

# Postmortem: Ingest Outage on March 12, 2024

Status: Final. Severity: SEV1. Incident commander and author: Jonah Weiss. Postmortem review held on March 19, 2024.

This is a blameless postmortem. Its purpose is to understand how our systems and processes allowed the outage, not to assign fault to individuals.

## Summary

On Tuesday, March 12, 2024, newly uploaded documents stopped becoming searchable for 3 hours and 42 minutes, from 14:05 to 17:47 UTC. Queries against documents that were already indexed kept working normally throughout. About 38% of active customers had at least one upload delayed during the window. No customer data was lost: every delayed document was indexed once the backlog drained.

## Timeline (all times UTC)

- 14:05: Riverbed v3.8.0 rolled out to 100% of workers in both regions at once.
- 14:31: The RiverbedLagHigh alert fired and paged the primary on-call engineer.
- 14:38: Jonah Weiss declared a SEV1 and took the incident commander role.
- 15:10: The team scaled Riverbed from 24 to 60 workers, which made the problem worse.
- 16:02: The team identified the consumer configuration change in v3.8.0 as the trigger.
- 16:20: Riverbed was rolled back to v3.7.4; the rollback itself took under 5 minutes.
- 17:47: The backlog finished draining and freshness returned to normal.

Consumer lag peaked at 2.1 million messages at about 16:15.

## Root cause

Riverbed v3.8.0 contained a configuration-only change that lowered the Kafka consumer setting max.poll.interval.ms from 300000 to 30000 milliseconds. The intent was to detect stuck workers faster. However, large scanned PDFs that need OCR routinely take longer than 30 seconds to process. Every time a worker spent more than 30 seconds on one document, Kafka assumed it was dead and removed it from the consumer group, which triggered a rebalance. During each rebalance no worker could make progress, so the lag kept growing.

Adding workers at 15:10 made things worse, because every new consumer joining the group triggered yet another rebalance.

## Why the change reached production

At the time of the incident, the release process exempted configuration-only changes from canary deploys. Because v3.8.0 changed only configuration, it went straight to every worker in both regions with no canary stage. The pull request was reviewed and approved, but the reviewer did not know that large PDFs often exceed 30 seconds of processing time.

## Detection

The RiverbedLagHigh alert threshold was 200,000 messages at the time of the incident, so the alert fired 26 minutes after the deploy. With a lower threshold the team would have been paged within a few minutes.

## Customer impact and credits

The customer support team contacted all 23 affected Enterprise customers on the same day. Each of them received a goodwill credit of 15% of their March invoice, approved by Priya Raman, even though query uptime was not affected.

## What went well

- Once the cause was identified, the rollback was fast and clean.
- No data was lost, because Kafka retained every event during the outage.
- Status page updates went out on schedule throughout the incident.

## What went poorly

- The alert threshold was too high to catch the problem early.
- Scaling up during a rebalance storm extended the outage.
- Nobody on the incident had a written list of Riverbed configuration changes in v3.8.0, so finding the trigger took almost two hours.

## Action items

1. Remove the canary exemption for configuration-only changes. Owner: Ravi Chandrasekar. Completed March 28, 2024.
2. Lower the RiverbedLagHigh alert threshold. Owner: Daniel Brandt. Completed March 15, 2024.
3. Add a warning to the On-Call Runbook against scaling Riverbed during rebalances. Owner: Marcus Lindqvist. Completed March 20, 2024.
4. Add a Riverbed integration test that processes a 5 MB scanned PDF with OCR and asserts it finishes within the poll interval. Owner: Jonah Weiss. Completed April 30, 2024.
5. Require every Riverbed release to include a generated list of configuration changes. Owner: Daniel Brandt. Completed April 12, 2024.
