---
allowed-tools: Read Bash Grep Glob
description: Comprehensive security review of code, configurations, and deployments to identify vulnerabilities and compliance gaps.
name: security-review
---

# Security Review Skill

Audit code and configuration for security vulnerabilities, compliance gaps, and attack surface.

## Trigger

Invoke this skill to identify security risks:
- Before deploying to production
- When onboarding a new dependency or service
- During architecture reviews for new systems
- After a security incident to prevent recurrence
- When handling sensitive data (PII, secrets, auth tokens)

## Usage

```
/security-review [--scope code|config|deployment] [--level quick|standard|deep]
```

### Scopes

- **code** — source code for OWASP Top 10, injection, logic bugs
- **config** — environment, credentials, permissions, TLS/cert settings
- **deployment** — infrastructure, access controls, observability, incident response

### Effort Levels

- **quick** — automated checks, obvious issues only (15 min)
- **standard** — manual code inspection, threat modeling (1 hour)
- **deep** — comprehensive audit, architecture review, pen-test prep (4+ hours)

## Dimensions Covered

### Code Security

- **Injection** — SQL, command, template, XSS, XXE, LDAP
- **Authentication** — credential storage, session management, MFA
- **Authorization** — privilege escalation, broken access control
- **Sensitive data** — hardcoded secrets, unencrypted transmission, logging PII
- **Dependencies** — known vulnerabilities, supply chain risk
- **Cryptography** — weak algorithms, insufficient randomness
- **Error handling** — information disclosure via stack traces

### Configuration Security

- **Secrets management** — env vars, vaults, rotation policies
- **Permission boundaries** — IAM policies, file permissions, network isolation
- **TLS/certificates** — expiry, pinning, HSTS headers
- **Logging** — what's logged, retention, access controls
- **Default credentials** — changed before deployment

### Deployment Security

- **Network** — firewall rules, egress filtering, DMZ topology
- **Identity** — service accounts, federated identity, key rotation
- **Observability** — audit logs, anomaly detection, alerting
- **Incident response** — runbooks, escalation, forensics capability
- **Backup/restore** — backup encryption, restore testing, SLA

## Anti-Patterns

- Don't ignore "low severity" findings on the assumption they won't be exploited
- Don't rely on obscurity as a security control
- Don't commit secrets even briefly (git history lives forever)
- Avoid security theater — invest in controls that meaningfully reduce risk

## Limitations

- Cannot penetration-test a live system (requires explicit authorization)
- Cannot audit compliance frameworks (SOC 2, HIPAA, PCI-DSS) in one pass
- Cannot verify correct runtime behavior without deployment context
- Cannot assess operational security practices (training, processes)

## Example: Code Security Review

```bash
git add .
/security-review --scope code --level standard
```

Reports injection risks, hardcoded secrets, weak auth patterns, and dependency vulnerabilities.
