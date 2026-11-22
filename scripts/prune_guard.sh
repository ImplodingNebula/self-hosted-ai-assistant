#!/bin/sh
# Refuse to prune volumes on a Docker engine where doing so destroys live data.
#
# WHAT THIS PROTECTS. `docker volume prune` removes ANONYMOUS volumes only -
# but that has been true since Docker Engine 23.0 (API 1.42), and NOT before.
# On an older daemon the same command removes every volume that no container is
# currently using, NAMED ONES INCLUDED. Stopping the stack is what makes that
# fatal: with the containers gone, `postgres_data`, `qdrant_data`,
# `attachment_data` and `ollama_models` are all "unused", so a routine cleanup
# deletes the database, the vector store, every uploaded file and the model
# weights in one command that reports success.
#
# There is no flag that restores the newer behaviour on an old daemon, so the
# project cannot make the command safe - it can only decline to run it and say
# why. That is what this does.
#
# WHY IT IS NOT HYPOTHETICAL. The project states no minimum engine version, and
# the gap is easy to land in: Ubuntu 22.04 still ships `docker.io` 20.10
# (API 1.41) alongside a current Compose plugin, which satisfies the stated
# prerequisites exactly. The targets that prune also advertise themselves as
# safe, so the operator has been told the opposite of the truth.
#
# BOTH ENDS HAVE TO CLEAR THE FLOOR, and checking only the daemon is the
# mistake that looks correct. Which behaviour a request gets is decided by the
# NEGOTIATED version - the lower of client and daemon - so an old client against
# a new daemon still asks for, and receives, the destructive one. Demonstrated
# against a single 1.55 daemon on this project's own machine: `docker volume
# prune --help` normally offers `-a/--all` (the flag that OPTS IN to removing
# named volumes, which only exists because the default no longer does), and with
# the client pinned to 1.41 that flag disappears entirely - the CLI is speaking
# the dialect where removing named volumes is not optional.
#
# Usage:
#   sh scripts/prune_guard.sh volumes            guard, then prune; fails if unsafe
#   sh scripts/prune_guard.sh volumes soft       guard, then prune; warns if unsafe
#   sh scripts/prune_guard.sh check 1.41         answer safe/unsafe for one version
#   sh scripts/prune_guard.sh check 1.55 1.41    ... or for a client/daemon pair
#
# The `check` form takes the versions as arguments so the comparison is testable
# without an old daemon to hand.

set -e

# Engine 23.0. Below this, `docker volume prune` includes named volumes.
MIN_MAJOR=1
MIN_MINOR=42

# Deliberately strict: anything unparseable is UNSAFE. A guard that treats an
# unreadable version as permission to continue is not a guard.
api_is_safe() {
	v="$1"
	case "$v" in
		*.*) ;;
		*) return 1 ;;
	esac
	maj=${v%%.*}
	min=${v#*.}
	min=${min%%.*}
	case "$maj" in ''|*[!0-9]*) return 1 ;; esac
	case "$min" in ''|*[!0-9]*) return 1 ;; esac

	if [ "$maj" -gt "$MIN_MAJOR" ]; then
		return 0
	fi
	if [ "$maj" -eq "$MIN_MAJOR" ] && [ "$min" -ge "$MIN_MINOR" ]; then
		return 0
	fi
	return 1
}

# Both must clear the floor. An unrecognised subcommand is an error rather than
# a silent fall-through to the pruning path - the default here deletes things.
MODE="${1:-volumes}"

case "$MODE" in
	volumes|check) ;;
	*)
		echo "prune_guard: unknown subcommand '$MODE'." >&2
		echo "  Expected: volumes [soft] | check <version> [version]" >&2
		exit 2
		;;
esac

if [ "$MODE" = "check" ]; then
	if api_is_safe "$2" && { [ -z "${3:-}" ] || api_is_safe "$3"; }; then
		echo "safe"
	else
		echo "unsafe"
	fi
	exit 0
fi

SOFT="${2:-}"

VERSIONS=$(docker version --format '{{.Server.APIVersion}} {{.Client.APIVersion}}' 2>/dev/null || true)
SRV=$(printf '%s' "$VERSIONS" | cut -d' ' -f1)
CLI=$(printf '%s' "$VERSIONS" | cut -d' ' -f2)

explain() {
	echo ""
	echo "  Removing unused volumes is limited to ANONYMOUS ones from Docker"
	echo "  Engine 23.0 (API 1.42) onward. Below that the same command also"
	echo "  removes NAMED volumes that nothing is currently using - which, with"
	echo "  the stack stopped, means the database, the vector store, the"
	echo "  uploaded files and the model weights."
	echo ""
	echo "  Which behaviour a request gets is decided by the LOWER of the client"
	echo "  and daemon versions, so an old client is enough on its own."
	echo ""
	echo "  No option restores the newer behaviour, so this stops instead."
	echo "  Either bring both sides to Docker 23.0 or newer, or reclaim only the"
	echo "  parts that carry no data:"
	echo "    docker image prune -f"
	echo "    docker builder prune -f"
}

soft_or_fail() {
	if [ "$SOFT" = "soft" ]; then
		echo ""
		echo "  Continuing: the rebuild itself succeeded, and the only cost is that"
		echo "  the orphaned anonymous volumes it left behind were not reclaimed."
		exit 0
	fi
	exit 1
}

if [ -z "$SRV" ] || [ -z "$CLI" ]; then
	echo "prune: could not ask Docker for its API version - is the daemon running?"
	echo "  Skipping the volume prune rather than assuming it is safe."
	soft_or_fail
fi

if ! api_is_safe "$SRV" || ! api_is_safe "$CLI"; then
	echo "prune: refusing to prune volumes on this Docker installation."
	echo "  daemon API $SRV, client API $CLI - both must be 1.42 or newer."
	explain
	soft_or_fail
fi

docker volume prune -f
