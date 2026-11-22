#!/bin/sh
# Report how much room the stack's data has, and how much of it is left.
# Invoked by `make host-disk`. Read-only: it asks Docker questions and changes
# nothing.
#
# WHY THIS IS A SEPARATE QUESTION FROM `make disk`. The two look alike and are
# not. `make disk` (scripts/wsl_disk.sh) is about the Windows virtual-disk FILE
# that never shrinks - a reclaim question, and one that exists only on Windows.
# This is the deployment question that applies everywhere: is the machine
# running out of space, and which of the stack's volumes is responsible.
#
# THE VOLUMES ARE THE PART THAT GROWS. Images and build cache reach a size and
# stay near it; data does not. Three volumes grow with use - the database, the
# uploaded files, the vectors - and one grows in steps as models are pulled.
# Only the first three are backed up, so the report says which is which:
# knowing a volume is large is only half of what an operator needs to decide
# what to do about it.
#
# Sizes come from the Docker daemon's own accounting (`docker system df -v`)
# rather than from walking the filesystem, which means no container has to be
# started and no volume has to be mounted to answer.

set -eu

ROOT=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)

project_name() {
	if [ -n "${COMPOSE_PROJECT_NAME:-}" ]; then
		printf '%s' "$COMPOSE_PROJECT_NAME"
		return 0
	fi
	sed -n 's/^name:[[:space:]]*//p' "$ROOT/infra/docker-compose.yml" | head -n 1 | tr -d '\r'
}

PROJECT=$(project_name)

# Consistent with scripts/backup.sh and scripts/restore.sh. Nothing here passes
# a container-side path today, so this changes no behaviour - it exists so an
# edit that adds one does not have to rediscover the Git Bash rewriting rule.
docker_() {
	MSYS_NO_PATHCONV=1 docker "$@"
}

if ! docker_ info >/dev/null 2>&1; then
	echo "make host-disk: cannot reach the Docker daemon - is Docker running?"
	exit 1
fi

DATA_ROOT=$(docker_ info --format '{{.DockerRootDir}}' 2>/dev/null || true)

echo "== Docker data root =="
if [ -z "$DATA_ROOT" ]; then
	echo "  Location   unknown   <- the daemon did not report one"
else
	echo "  Location   $DATA_ROOT"
fi

# The data root is a path on whichever machine the DAEMON runs on. On Linux
# that is this machine, so `df` answers about the real filesystem. Under Docker
# Desktop the daemon lives in a virtual machine and the path does not exist
# here - reporting the closest local mount instead would be a confident wrong
# number, so say what the situation is and point at the tool that does apply.
if [ -n "$DATA_ROOT" ] && [ -d "$DATA_ROOT" ]; then
	echo ""
	echo "  Filesystem holding it:"
	df -Ph "$DATA_ROOT" | sed 's/^/    /'
	echo ""
	echo "  Everything below lives on that filesystem. When it fills, Postgres"
	echo "  stops accepting writes and model pulls fail part-way - so keep an eye"
	echo "  on the used percentage rather than on any single volume."
else
	echo ""
	echo "  The daemon runs inside a virtual machine on this host, so its data root"
	echo "  is not a directory here and free space cannot be read from it directly."
	echo "  On Windows, 'make disk' reports the virtual-disk file that backs it."
fi

echo ""
echo "== Data volumes for project '$PROJECT' =="
echo ""

# `docker system df -v` prints several tables; the volume one starts at its
# header and ends where the next table begins. Matched on the header rather
# than on a line count, which changes with how many images are present.
VOLUMES=$(docker system df -v 2>/dev/null | awk '
	/^VOLUME NAME/ { inblock = 1; next }
	inblock && /^[[:space:]]*$/ { inblock = 0 }
	inblock && /^(CACHE ID|Build cache)/ { inblock = 0 }
	inblock { print }
' || true)

if [ -z "$VOLUMES" ]; then
	echo "  Docker reported no volumes."
	exit 0
fi

# What each volume is for, and whether `make backup` copies it. Kept beside the
# sizes on purpose: "qdrant_data is 12 GB" and "qdrant_data is backed up" are
# the two facts an operator needs at the same moment.
describe() {
	case "$1" in
	*_postgres_data) echo "backed up     conversations, messages, attachment rows, credentials" ;;
	*_attachment_data) echo "backed up     uploaded file bytes" ;;
	*_qdrant_data) echo "backed up     retrieval vectors" ;;
	*_redis_data) echo "not backed up job queue only; losing it loses queued work, no records" ;;
	*_ollama_models) echo "not backed up model weights; re-downloadable from the model manager" ;;
	*_infinity_cache) echo "not backed up reranker weights; re-downloaded on first start" ;;
	*_minio_data) echo "not backed up self-hosted object storage; mirror or copy it cold" ;;
	*_caddy_data) echo "not backed up TLS certificates and keys; re-issued on demand" ;;
	*_caddy_config) echo "not backed up the proxy's last accepted configuration" ;;
	*) echo "" ;;
	esac
}

echo "$VOLUMES" | while IFS= read -r line; do
	name=$(printf '%s' "$line" | awk '{print $1}')
	size=$(printf '%s' "$line" | awk '{print $NF}')
	case "$name" in
	"$PROJECT"_*) ;;
	*) continue ;;
	esac
	short=${name#"$PROJECT"_}
	printf '  %-18s %10s   %s\n' "$short" "$size" "$(describe "$name")"
done

# A volume that has never been created does not appear above, which is normal
# on a stack that has not started every profile - so the absence of a row is
# not a fault and is not reported as one.
echo ""
echo "  Sizes are the Docker daemon's own accounting. A volume missing from this"
echo "  list has not been created yet (an opt-in profile, or a store that has"
echo "  never started)."

# The backups directory is the one thing this stack grows that is NOT a Docker
# volume, and nothing rotates it - every `make backup` adds a full copy. Left
# out of the report it would be invisible to the only tool that watches disk,
# which is precisely where an unnoticed pile of copies fills a server.
echo ""
echo "== Local backups =="
echo ""
if [ -d "$ROOT/backups" ]; then
	echo "  $ROOT/backups"
	printf '  %s total, %s run(s)\n' \
		"$(du -sh "$ROOT/backups" 2>/dev/null | cut -f1)" \
		"$(find "$ROOT/backups" -mindepth 1 -maxdepth 1 -type d 2>/dev/null | wc -l | tr -d ' ')"
	echo ""
	echo "  Nothing prunes these. Each 'make backup' writes another full copy, so"
	echo "  delete the runs that are no longer wanted, or send new ones elsewhere"
	echo "  with 'make backup DEST=<dir>' - ideally onto a different disk."
else
	echo "  None here yet. 'make backup' writes to $ROOT/backups unless DEST says"
	echo "  otherwise, and nothing prunes what accumulates there."
fi
echo ""
echo "  'make backup' copies the three volumes marked above; see docs/DATA-MODEL.md,"
echo "  'What lives in each volume', for why the rest are excluded."
