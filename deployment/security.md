# Security Measures

## Sensitive Data Handling

- **Secrets Management:** All API keys, credentials, and tokens are stored as environment variables or in GitHub/Streamlit secrets—not in code.
- **Access Control:** Principle of least privilege for all user/service accounts.
- **Data Encryption:** All sensitive data (datasets, models) is encrypted at rest and in transit.
- **Audit Logging:** Access to sensitive endpoints and data is logged for auditing.

## Application Security

- **Input Validation:** All API endpoints validate and sanitize input.
- **CSRF/XSS Protection:** Use secure frameworks for API and UI.
- **Dependency Management:** `pip-audit` and Dependabot monitor for vulnerable packages.

## Incident Response

- **Alerts:** Security events trigger email/SMS alerts.
- **Patch Management:** Critical patches applied within 24 hours.
- **Documentation:** Security policies and incident response plans are in `docs/wiki.md`.

---

**Contact admin if you discover a vulnerability.**