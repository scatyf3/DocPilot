> Fictional document for the DocPilot eval corpus. Lumen Labs, Lumen Search, and every person named here are invented.

# Information Security Policy

Document ID: LL-SEC-01. Owner: Fatima Al-Sayed, Head of Security. Version 4, effective January 15, 2025.

This policy applies to every employee, contractor, and intern at Lumen Labs, and to every system that stores or processes company or customer data. Violations may lead to loss of access or disciplinary action. Questions about this policy go to the #ask-security channel.

## Identity and access

All employees sign in through Okta single sign-on with multi-factor authentication. SMS and voice codes are not accepted as a second factor anywhere in the company. Production access additionally requires a YubiKey hardware security key.

Engineers do not hold standing production access. Instead, they request just-in-time access through the Gatekeeper tool, and each request must be approved by a second engineer. Gatekeeper sessions expire automatically after 4 hours, and every command run during a session is recorded.

Managers review the access of everyone on their team once per quarter. When someone leaves the company, all of their access must be revoked within 2 hours of their departure being confirmed by People Operations.

For the rare systems that cannot use single sign-on, passwords must be at least 14 characters long and must be stored in the company 1Password vault, never in a browser or a document.

## Secrets management

Application secrets live in HashiCorp Vault and are injected at runtime. Secrets must never be committed to git; a gitleaks pre-commit hook and a CI check block commits that look like credentials. The JWT signing keys used by Keystone are rotated every 90 days, with the previous key kept valid for one extra day so that existing tokens do not break. Customer API keys can be rotated by customers at any time, and keys that have not been used for 180 days are flagged to the customer for removal.

## Devices

Company laptops must use full-disk encryption with FileVault and must lock the screen after 5 minutes of inactivity. All laptops are enrolled in the Kandji device management system, which enforces these settings. Operating system security updates must be installed within 14 days of release. Personal devices may be used for Slack and email only, and never for production access.

## Vulnerability management

Dependabot scans every repository for vulnerable dependencies, and every container image is scanned with Trivy before deployment. Vulnerabilities must be fixed within the following deadlines, measured from when they are confirmed:

- Critical vulnerabilities must be fixed within 7 days.
- High vulnerabilities must be fixed within 30 days.
- Medium vulnerabilities must be fixed within 90 days.
- Low vulnerabilities are fixed on a best-effort basis and tracked in the backlog.

## Change review

Any change to Keystone, including configuration-only changes, requires approval from a member of the Security team before it can be merged. This rule was added after the July 2024 auth latency incident. Changes to Beacon rate-limiting rules need the same Security approval.

## Vendors

New SaaS vendors that will hold company or customer data need a vendor security review before anyone signs a contract or enters a credit card. The Security team usually completes a vendor review within 10 business days. Vendors that will process customer documents must also sign a data processing agreement reviewed by Legal.

## Reporting security incidents

Anyone who suspects a security incident, including a lost laptop or YubiKey, must report it to security@lumenlabs.example or in #ask-security within 1 hour of discovery. Do not try to investigate on your own first. The on-call engineer and the Security team classify the severity together, and the Head of Security decides whether customers or regulators must be notified.

## Audits and testing

Lumen Labs completes a SOC 2 Type II audit every year, with an observation window that ends on September 30. An external penetration test is performed once a year, during the second quarter, by the firm Ironclad Assurance. Every employee receives a simulated phishing email once per quarter; anyone who fails two simulations in a row must retake the security awareness training.

## Exceptions

Exceptions to this policy must be approved in writing by the Head of Security. An exception lasts at most 6 months and must be renewed explicitly if it is still needed.
