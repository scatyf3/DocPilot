> Fictional document for the DocPilot eval corpus. Lumen Labs, Lumen Search, and every person named here are invented.

# Data Retention Policy

Document ID: LL-LEG-03. Owner: Carlos Ibáñez, General Counsel and Data Protection Officer. Effective October 1, 2024.

This policy sets how long Lumen Labs keeps each kind of data and how it is deleted. The goal is to keep data only as long as it is useful to customers or required by law. Engineers who build a new data store must add it to this policy before the store goes to production.

## Customer content

Customer documents and indexes are kept for as long as the customer's account is active. After an account is closed, the customer's documents, passages, vectors, and index configuration are deleted from primary storage within 30 days. Customers can export their documents at any time before closure through the export endpoint.

Daily snapshots of Lattice and meta-db are kept for 35 days and then expire automatically. Because snapshots are never edited, deleted customer content disappears from backups once the last snapshot containing it expires.

When a person exercises their right to erasure under GDPR, Lumen Labs completes the deletion within 21 days of verifying the request. Erasure requests are tracked by the Legal team in a dedicated Jira project named PRIV.

## EU data residency

For customers on EU hosting, all customer content, including snapshots, stays in the eu-central-1 region. Support engineers outside the EU may view metadata about an EU customer's indexes but may not download or view the documents themselves.

## Operational data

- Query logs, which contain the query text, index ID, and latency, are kept for 90 days. After 90 days they are rolled up into anonymized statistics and the query text is removed.
- Application logs are kept in Loki for 14 days and then archived to S3, where they are kept for another 180 days before deletion.
- Distributed traces are kept for 7 days.
- Audit logs, which record administrative actions and every Gatekeeper production session, are kept for 7 years.
- Incident postmortems are kept indefinitely, because they are the company's memory of what went wrong.

## Employee and business records

- Personnel files are kept for 6 years after employment ends.
- Financial records, including expense reports and receipts, are kept for 7 years.
- Applications from candidates who were not hired are kept for 1 year, unless the candidate asks for earlier deletion.
- Slack messages in public channels are kept for 3 years; direct messages are kept for 1 year.

## Legal hold

The General Counsel may place a legal hold on any data related to a dispute or investigation. A legal hold suspends every deletion rule in this policy for the covered data until the hold is lifted. The Infrastructure team must implement a new legal hold within 2 business days of receiving it.

## Deletion methods

Data in S3 is deleted using lifecycle rules, and deletions from Lattice are verified by a nightly job that compares index contents against the list of closed accounts. Laptops that are retired are wiped by the Kandji device management system before being recycled.

## Review

This policy is reviewed every October by the General Counsel together with the Head of Security. Any change to a retention period requires approval from the CEO.
