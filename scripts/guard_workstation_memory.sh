#!/usr/bin/env bash
# Preempt explicitly listed auxiliary tmux jobs before a workstation user cgroup reaches OOM.

set -euo pipefail

MEMORY_GUARD_GIB="${MEMORY_GUARD_GIB:-330}"
POLL_SECONDS="${POLL_SECONDS:-10}"
SESSIONS_TO_PREEMPT="${SESSIONS_TO_PREEMPT:-}"
GUARD_LOG="${GUARD_LOG:-${PWD}/logs/workstation-memory-guard.log}"

if ! [[ "${MEMORY_GUARD_GIB}" =~ ^[1-9][0-9]*$ ]]; then
    echo "MEMORY_GUARD_GIB must be a positive integer" >&2
    exit 2
fi
if ! [[ "${POLL_SECONDS}" =~ ^[1-9][0-9]*$ ]]; then
    echo "POLL_SECONDS must be a positive integer" >&2
    exit 2
fi
if [[ -z "${SESSIONS_TO_PREEMPT}" ]]; then
    echo "SESSIONS_TO_PREEMPT must list at least one tmux session" >&2
    exit 2
fi

user_id="$(id -u)"
cgroup="/sys/fs/cgroup/user.slice/user-${user_id}.slice/user@${user_id}.service"
memory_current="${cgroup}/memory.current"
threshold_bytes=$((MEMORY_GUARD_GIB * 1024 * 1024 * 1024))

if [[ ! -r "${memory_current}" ]]; then
    echo "cannot read user-cgroup memory counter: ${memory_current}" >&2
    exit 2
fi

mkdir -p "$(dirname "${GUARD_LOG}")"
exec >>"${GUARD_LOG}" 2>&1

session_is_active() {
    local session=$1
    tmux list-panes -t "${session}" -F '#{pane_dead}' 2>/dev/null | grep -qx '0'
}

echo "[$(date -Is)] memory guard armed at ${MEMORY_GUARD_GIB} GiB; preemption order: ${SESSIONS_TO_PREEMPT}"
while true; do
    current_bytes="$(<"${memory_current}")"
    if (( current_bytes >= threshold_bytes )); then
        preempted=0
        for session in ${SESSIONS_TO_PREEMPT}; do
            if session_is_active "${session}"; then
                echo "[$(date -Is)] usage=$((current_bytes / 1024 / 1024 / 1024)) GiB; preempting ${session}"
                tmux kill-session -t "${session}" 2>/dev/null || true
                preempted=1
                break
            fi
        done
        if (( preempted == 0 )); then
            echo "[$(date -Is)] WARNING usage=$((current_bytes / 1024 / 1024 / 1024)) GiB; no listed auxiliary session is active"
        fi
    fi
    sleep "${POLL_SECONDS}"
done
