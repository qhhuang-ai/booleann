#!/usr/bin/env bash
set -euo pipefail

root=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
status=0

while IFS= read -r path; do
  printf 'unexpected external-code directory: %s\n' "$path" >&2
  status=1
done < <(
  find "$root" -mindepth 1 -type d \
    \( -iname parlayann -o -iname sieve -o -iname sieve-vldb25 \
       -o -iname acorn -o -iname serf -o -iname wow \
       -o -iname third_party -o -iname vendor -o -name .git \) \
    ! -path "$root/.git" -print
)

while IFS= read -r path; do
  printf 'unexpected symlink: %s\n' "$path" >&2
  status=1
done < <(find "$root" -type l -print)

while IFS= read -r path; do
  printf 'unexpected patch or upstream license file: %s\n' "$path" >&2
  status=1
done < <(
  find "$root" -type f \
    \( -iname '*.patch' -o -iname '*.diff' -o -iname 'copying*' \
       -o -iname 'license*' \) ! -path "$root/LICENSE" -print
)

if (( status != 0 )); then
  exit "$status"
fi

printf 'release boundary: PASS (project source plus external references only)\n'
