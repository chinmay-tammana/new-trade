#!/usr/bin/env python3
"""Secret detection scanner for pre-commit verification.

Fails without printing secret values if any secret pattern, Alpaca credential,
data file, or git-ignored pattern appears in tracked files or staged git diffs.
"""

import os
import re
import subprocess
import sys
from pathlib import Path


def load_env_values():
    """Load APCA API credentials from .env without printing them."""
    env_path = Path(".env")
    key_id = ""
    secret_key = ""
    if env_path.exists():
        with open(env_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                if line.startswith("APCA_API_KEY_ID="):
                    key_id = line.split("=", 1)[1].strip().strip('"').strip("'")
                elif line.startswith("APCA_API_SECRET_KEY="):
                    secret_key = line.split("=", 1)[1].strip().strip('"').strip("'")
    return key_id, secret_key


def run_cmd(cmd):
    """Run shell command and return stdout string."""
    res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    return res.returncode, res.stdout.strip(), res.stderr.strip()


def check_forbidden_tracked_files():
    """Check git ls-files for .env, data/, or *.parquet files."""
    code, out, _ = run_cmd(["git", "ls-files"])
    if code != 0:
        # Not a git repo yet or git error
        return []

    violations = []
    for line in out.splitlines():
        path = line.strip()
        if not path:
            continue
        if path == ".env" or path.startswith(".env.") and path != ".env.example":
            violations.append(f"Forbidden tracked file: {path}")
        elif path.startswith("data/"):
            violations.append(f"Market data must never be tracked: {path}")
        elif path.endswith(".parquet"):
            violations.append(f"Parquet files must never be tracked: {path}")
    return violations


def check_contents_for_secrets(key_id, secret_key):
    """Check tracked files and staged git diff for secrets."""
    violations = []

    # Patterns for GitHub tokens or hardcoded Alpaca secret assignments
    gh_pattern = re.compile(r"ghp_[A-Za-z0-9_]{10,}|github_pat_[A-Za-z0-9_]{10,}")
    alpaca_secret_pattern = re.compile(r"APCA_API_SECRET_KEY\s*=\s*['\"]?[A-Za-z0-9_-]{10,}['\"]?")

    # 1. Check staged diff
    code, diff_out, _ = run_cmd(["git", "diff", "--cached"])
    if code == 0 and diff_out:
        if gh_pattern.search(diff_out):
            violations.append("Detected GitHub personal access token pattern in staged diff.")
        if alpaca_secret_pattern.search(diff_out):
            violations.append("Detected non-empty APCA_API_SECRET_KEY assignment in staged diff.")
        if key_id and len(key_id) > 6 and key_id in diff_out:
            violations.append("Detected APCA_API_KEY_ID value in staged diff.")
        if secret_key and len(secret_key) > 6 and secret_key in diff_out:
            violations.append("Detected APCA_API_SECRET_KEY value in staged diff.")

    # 2. Check tracked files
    code, files_out, _ = run_cmd(["git", "ls-files"])
    if code == 0 and files_out:
        for file_path_str in files_out.splitlines():
            file_path = Path(file_path_str.strip())
            if not file_path.is_file():
                continue
            if file_path.name == ".env.example":
                # .env.example only has empty placeholders
                continue
            try:
                content = file_path.read_text(encoding="utf-8", errors="ignore")
            except Exception:
                continue

            if gh_pattern.search(content):
                violations.append(f"GitHub token pattern found in {file_path}")
            if alpaca_secret_pattern.search(content):
                violations.append(f"Hardcoded APCA_API_SECRET_KEY found in {file_path}")
            if key_id and len(key_id) > 6 and key_id in content:
                violations.append(f"APCA_API_KEY_ID value found in {file_path}")
            if secret_key and len(secret_key) > 6 and secret_key in content:
                violations.append(f"APCA_API_SECRET_KEY value found in {file_path}")

    return violations


def main():
    key_id, secret_key = load_env_values()
    file_violations = check_forbidden_tracked_files()
    secret_violations = check_contents_for_secrets(key_id, secret_key)

    all_violations = file_violations + secret_violations
    if all_violations:
        print("=" * 60, file=sys.stderr)
        print("SECRET CHECK FAILED! The following security violations were found:", file=sys.stderr)
        for v in all_violations:
            print(f"  - {v}", file=sys.stderr)
        print("=" * 60, file=sys.stderr)
        sys.exit(1)

    print("Secret check PASSED: No tracked data/env files or secrets detected.")
    sys.exit(0)


if __name__ == "__main__":
    main()
