> Fictional document for the DocPilot eval corpus. Lumen Labs, Lumen Search, and every person named here are invented.

# On-Call Runbook

Owner: Infrastructure team. Last reviewed: January 6, 2025.

This runbook explains how the Lumen Search on-call rotation works, how incidents are classified, and what to do for the most common alerts. Pay for on-call shifts is covered in the Benefits and PTO guide, not here.

## Rotation

The rotation is weekly and hands off every Monday at 10:00 Pacific time. Each week has a primary and a secondary on-call engineer, drawn from the Infrastructure, Ingest, Search Relevance, Identity, and API Platform teams. The outgoing primary writes a handoff note in #oncall-handoff summarizing open alerts and ongoing incidents. You may swap shifts with a colleague as long as you update the PagerDuty schedule yourself at least 48 hours before the shift starts.

## Severity levels

- SEV1: a full outage of the query path or the ingest path, confirmed loss of customer data, or a security incident that exposes customer data.
- SEV2: significant degradation, such as a query error rate above 2% for 10 minutes or more, or one region unavailable while the other region stays healthy.
- SEV3: minor impact with a known workaround, such as a single customer's index rebuilding slowly.

## Acknowledgement and escalation

A SEV1 page must be acknowledged within 5 minutes, and a SEV2 page within 15 minutes. SEV3 issues are handled during business hours and never page anyone at night.

If the primary does not acknowledge a page within 5 minutes, PagerDuty automatically pages the secondary. If neither engineer has acknowledged after 15 minutes, PagerDuty escalates to Marcus Lindqvist as the final escalation contact. For any SEV1 that lasts longer than 30 minutes, the incident commander must also notify Helen Vasquez directly by phone.

## Running an incident

### Declaring

Declare an incident by typing /incident declare in Slack. The bot creates a dedicated incident channel and posts a link to it in #inc-war-room. For SEV1 incidents, an incident commander must be named within 10 minutes of the declaration.

### The incident commander

Incident commanders must be level E3 or above and must have shadowed at least two incidents. The incident commander coordinates the response, delegates investigation, and keeps the timeline; they should not also be the person typing the fix.

### Customer communication

For SEV1 and SEV2 incidents, post an initial update to the public status page at status.lumensearch.example within 20 minutes of declaring, and then post a new update every 30 minutes until the incident is resolved. The customer support team mirrors these updates to affected Enterprise customers.

### Postmortems

Every SEV1 and SEV2 incident requires a blameless postmortem, published within 5 business days of resolution. The incident commander is responsible for scheduling the postmortem review meeting and assigning an owner to every action item.

## Common alerts

### PrismP99High

This alert fires when Prism query latency at p99 stays above 800 ms for 5 consecutive minutes. First open the Grafana dashboard named Prism Query Latency and look for a single hot index, because one customer's runaway query often explains the spike. If latency is high across all indexes, check whether a deploy happened in the last hour and roll it back.

### RiverbedLagHigh

This alert fires when Riverbed consumer lag exceeds 50,000 messages on the ingest.raw.v3 topic. Check the Riverbed logs for consumer group rebalances before you add workers. Never scale Riverbed workers up during a rebalance storm, because new consumers joining the group trigger even more rebalances.

### KeystoneLatencyP99

This alert fires when Keystone token validation latency at p99 exceeds 250 ms for 3 minutes. Check meta-db CPU first and confirm that the signing-key cache hit rate is above 99%. The July 2024 auth latency postmortem describes the failure mode this alert was created for.

### KeystoneErrorRate

This alert fires when more than 1% of Keystone requests return 5xx errors over a 5-minute window. A burst of errors right after a deploy is almost always a bad release, so roll back first and investigate second.

## Rolling back

To roll back a service, run lumenctl rollback with the --service flag and the service name. The command points Argo CD at the previous healthy revision and waits for the rollout to finish, which normally takes 2 to 4 minutes. Rollbacks do not need anyone's approval during an active incident. After rolling back, post the reverted version number in the incident channel.

## Useful links

- PagerDuty schedules: Lumen Primary and Lumen Secondary.
- Grafana: grafana.internal.lumenlabs.example.
- Error tracking: the Sentry project named lumen-prod.
- Status page admin: ask Infrastructure for access; only incident commanders should post.
