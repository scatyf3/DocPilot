> Fictional document for the DocPilot eval corpus. Lumen Labs, Lumen Search, and every person named here are invented.

# New Hire Onboarding Guide

Welcome to Lumen Labs. This guide walks through your first 90 days as an engineer working on Lumen Search, our hosted document-search API. Engineering onboarding is run by Nadia Petrova, and general questions can go to the #people-ops Slack channel at any time.

## Before your first day

Your laptop is shipped to your home address five business days before your start date. Engineers receive a 14-inch MacBook Pro with 32 GB of RAM; if you need a different configuration, tell IT at least two weeks before you start. You will also receive a one-time home office stipend, which is described in the Benefits and PTO guide.

## Day one checklist

- Log in to Okta first, because every other tool (Slack, GitHub, Google Workspace, PagerDuty, Grafana) uses Okta single sign-on.
- Register your YubiKey hardware security key, which arrives in the same box as your laptop.
- Join the #eng-announcements, #release-train, and #inc-war-room channels.
- Meet your onboarding buddy, a peer from your own team who is assigned before you arrive and meets with you three times a week during your first month.
- Read the PTO policy, document LL-HR-07, which explains how time off accrues and how to request it.

## Week one

### Required training

All new hires must complete the security awareness training within 5 business days of their start date. The course takes about 90 minutes and lives in the Learnwell portal. Production access is not granted until the training is marked complete.

### Setting up your development environment

Clone the monorepo, lumen-platform, from GitHub, and install the internal CLI with Homebrew. Running lumenctl dev up starts every service locally in Docker and seeds a sample index called idx_demo with 5,000 documents. A full local start usually takes about 4 minutes on a new laptop. If it takes longer than 15 minutes, ask in #dev-productivity.

### Your first pull request

Your goal for week one is to merge your first pull request. Most people pick a ticket labeled good-first-issue in the Jira project LUM. Your buddy reviews this first change, and it ships through the normal release process like any other change.

## The 30/60/90 day plan

### Days 1 to 30

Learn the architecture, starting with the Architecture Overview. Ship at least three small changes, and attend one API Review Board meeting as an observer to see how public API changes are discussed.

### Days 31 to 60

Shadow the on-call rotation for one full week during your second month. Shadowing means you receive the same pages as the primary on-call engineer but are not expected to respond on your own. Also pair with your buddy on at least one feature that touches two services.

### Days 61 to 90

Own a small project end to end, including writing a short design doc. Engineers join the regular on-call rotation after 90 days, once their manager confirms the shadow week is complete.

## Check-ins

Your manager holds a 30-minute check-in at the end of day 30, day 60, and day 90. At the 90-day check-in, you and your manager agree on goals that will feed into your first promotion cycle; the Engineering Ladder explains how levels and promotions work.

## Equipment and accounts after onboarding

Laptops are refreshed every three years. If you lose a laptop or YubiKey, report it to Security within one hour, the same deadline as any other security incident. When you leave the company, all equipment must be returned within 10 business days using the prepaid box that IT sends you.

## Getting help

- Tooling or local environment problems: #dev-productivity, owned by Ravi Chandrasekar's team.
- Benefits, payroll, and PTO questions: Grace Mbeki in People Operations.
- Anything that looks like a security issue: email security@lumenlabs.example right away.
- Questions about the onboarding program itself: Nadia Petrova.
