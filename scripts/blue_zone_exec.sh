#!/usr/bin/env bash
# Reuse an existing host ControlMaster; execute argv inside the research container.
set -euo pipefail

ssh_host="${BLUE_ZONE_SSH_HOST:-root@116.204.40.238}"
ssh_config="${BLUE_ZONE_SSH_CONFIG:-/dev/null}"
container="${BLUE_ZONE_CONTAINER:-research_vllm_ascend_021_external_workspace}"
control_path="${BLUE_ZONE_CONTROL_PATH:-}"
ssh_args=(ssh -F "$ssh_config" -o BatchMode=yes -o ControlMaster=no -o ProxyCommand=false)

if [[ -z "$control_path" ]]; then
  # Linux orchestration masters are kept in this per-user directory. A Mac
  # master can instead use the scope-%C path or an explicit override.
  shopt -s nullglob
  candidates=(/tmp/codex-blue-zone-"$(id -u)"/* /tmp/codex-blue-zone-scope-%C)
  for candidate in "${candidates[@]}"; do
    if "${ssh_args[@]}" -S "$candidate" -O check "$ssh_host" >/dev/null 2>&1; then
      control_path="$candidate"
      break
    fi
  done
fi
if [[ -z "$control_path" ]]; then
  echo 'No live Blue Zone host master found. Set BLUE_ZONE_CONTROL_PATH to the existing host socket.' >&2
  exit 1
fi
"${ssh_args[@]}" -S "$control_path" -O check "$ssh_host"
if [[ "${1:-}" == --check ]]; then
  echo "BLUE_ZONE_CONTROL_PATH=$control_path"
  exit 0
fi
if [[ $# == 0 ]]; then
  echo 'Usage: bash scripts/blue_zone_exec.sh COMMAND [ARG ...]' >&2
  exit 2
fi
# %q preserves argv through the host's shell; there is no eval or implicit shell
# inside the container. Ask for bash explicitly when sourcing NPU setup.
printf -v remote_command '%q ' docker exec -i "$container" "$@"
exec "${ssh_args[@]}" -S "$control_path" "$ssh_host" "$remote_command"
