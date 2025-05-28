# Backup & Disaster Recovery

## Backup Procedures

- **Daily Backups:** All critical datasets and model files are backed up automatically every night to encrypted cloud storage (e.g., AWS S3, Google Drive).
- **Versioning:** Each backup is timestamped for easy retrieval of historical data.
- **Database Backups:** If using a database, use built-in dump/export tools (e.g., `pg_dump` for PostgreSQL).

## Disaster Recovery

- **Restore Instructions:** Clear documentation for restoring from backup is included in this repo.
- **Testing:** Periodically test restoration to ensure backup integrity.
- **Redundancy:** Use multiple storage providers if possible.

## Example Backup Script

```bash
#!/bin/bash
tar -czf backup_$(date +%Y%m%d).tar.gz datasets/ models/
rclone copy backup_$(date +%Y%m%d).tar.gz remote:climate-backups/
```
---

**See `docs/maintenance.md` for full recovery walkthrough.**