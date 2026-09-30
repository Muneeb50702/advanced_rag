# Backup and restore

The application database is backed up every six hours. Backups are encrypted at rest and retained for thirty days. Store backup credentials separately from the backup files.

Run a restore drill once per quarter in an isolated environment. Verify row counts and application health before marking the drill complete. Never restore a backup over production without an approved recovery plan.
