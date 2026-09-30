# Deployment procedure

Deployments require a passing test suite and a reviewed change. The release owner records the commit SHA and the planned rollback version before promotion.

Promote to staging first. Run the smoke test against the staging health endpoint, then promote the same artifact to production. Watch error rate and latency for twenty minutes after release.

If the error rate exceeds the release threshold, roll back to the recorded version and notify the incident channel. Do not patch the running container by hand.
