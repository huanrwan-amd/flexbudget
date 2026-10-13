"""Content validator for a re-run result JSON. Exit 0 = genuinely valid.

An mtime check is NOT sufficient. A job can rewrite the JSON with zero rows and
elapsed_sec=0.12 after every model import failed; that looks "fresh" to a timestamp
check but is an empty/mock record. This enforces the project's factual-audit gates:
real GPU, real elapsed time, no recorded errors, non-empty payload.
"""
import json
import sys

path = sys.argv[1]
try:
    d = json.load(open(path))
except Exception as exc:  # unreadable or truncated
    print(f"INVALID: unreadable ({exc})")
    sys.exit(1)

env = d.get("env", {}) or {}
if not env.get("cuda_available"):
    print("INVALID: cuda_available is not true (CPU fallback or no GPU)")
    sys.exit(1)
if "MI300X" not in str(env.get("device_name", "")):
    print(f"INVALID: unexpected device {env.get('device_name')!r}")
    sys.exit(1)

elapsed = d.get("elapsed_sec") or 0
if elapsed < 1.0:
    print(f"INVALID: elapsed_sec={elapsed} (<1s implies a mock or failed run)")
    sys.exit(1)

errors = d.get("errors") or {}
if errors:
    first = str(list(errors)[0])[:80]
    print(f"INVALID: {len(errors)} recorded error(s), e.g. {first}")
    sys.exit(1)

payload = (
    len(d.get("rows", []) or [])
    or len(d.get("length_sweep", {}) or {})
    or len(d.get("parity", {}) or {})
)
if payload == 0:
    print("INVALID: empty payload (no rows / length_sweep / parity)")
    sys.exit(1)

print(f"VALID: elapsed={elapsed}s payload={payload} device={env.get('device_name')}")
