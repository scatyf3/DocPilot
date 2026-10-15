> Fictional document for the DocPilot eval corpus. Lumen Labs, Lumen Search, and every person named here are invented.

# Release Process

Owner: Ravi Chandrasekar, release manager. Version 2.3, last updated September 16, 2024.

This document describes how code gets from a merged pull request to production for every Lumen Search service. If anything here conflicts with an instruction in a team's own README, this document wins.

## Overview

All production services are built by GitHub Actions and deployed by Argo CD from the main branch. Argo Rollouts manages canary traffic. Each service is versioned independently with semantic versioning, and every release is a git tag of the form vMAJOR.MINOR.PATCH, for example riverbed v3.7.4.

## Before you merge

- CI must be green, and the pull request needs at least one approving review.
- Unit test line coverage for the changed package must stay at or above 75%.
- Every user-visible change needs an entry under the Unreleased heading in the service's CHANGELOG.md.
- Database migrations must be backward compatible with the previous release, so that a rollback never requires a schema rollback.

## Deploy windows

Production deploys are allowed Monday through Thursday between 09:00 and 15:00 Pacific time. Deploys on Fridays, weekends, or US public holidays require written approval from the VP of Engineering. Every year there is a code freeze from December 18 through January 4, during which only fixes for SEV1 or SEV2 incidents may be deployed.

## Canary deploys

Every production deploy starts as a canary that receives 5% of traffic for 30 minutes. Argo Rollouts automatically rolls the canary back if its error rate exceeds 1.5 times the baseline error rate, or if its p99 latency exceeds 1.3 times the baseline latency. If the canary is healthy after 30 minutes, traffic shifts to 100% in a single step.

Since March 2024, configuration-only changes go through the same canary as code changes. Before that, configuration-only changes were exempt from canary, and the exemption was removed after the March 2024 ingest outage.

## Load testing on the authentication path

Since August 2024, any release of a service on the authentication path, which today means Keystone and Beacon, must pass the k6 auth-soak test before it is promoted past canary. The auth-soak test runs for 20 minutes at 3 times the previous week's peak traffic against a staging copy of meta-db. A release fails the test if p99 validation latency rises above 100 ms at any point.

## Hotfixes

A hotfix branches from the latest release tag rather than from main. Hotfixes need two approving reviews, and at least one of them must come from an engineer on the Infrastructure team. Hotfixes may be deployed outside the normal deploy windows, but they may not skip the canary; for hotfixes the canary stage is shortened to 10 minutes.

## Release communication

On every deploy day, the release manager posts a summary of what shipped in #release-train by 16:00 Pacific. Changes that customers can see in the public API also get public release notes on the documentation site within 2 business days of reaching production.

## Rollbacks

Any engineer may roll back their own service at any time without approval, and the On-Call Runbook describes the rollback command. A rollback does not reset the deploy window: if you roll back at 14:50 on a Thursday, the fixed version waits until Monday unless you get approval.

## Release metrics

Developer Productivity tracks four metrics for every service and reviews them monthly with each team lead:

- Deployment frequency, with a target of at least 3 production deploys per week per active service.
- Lead time from merge to production, with a target under 24 hours.
- Change failure rate, meaning the share of deploys that are rolled back, with a target under 5%.
- Time to restore after a failed deploy, with a target under 30 minutes.
