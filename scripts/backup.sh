#!/bin/sh
# Take a consistent, restorable copy of everything this stack cannot re-create.
# Invoked by `make backup`. Reads the running stack; writes only into the
# destination directory.
#
# WHAT IS WORTH COPYING, and why the three artifacts are not symmetrical.
# The stack keeps seven named volumes and only three of them hold anything a
# reinstall could not reproduce:
#
#   postgres_data     every conversation, message, attachment row, and the
#                     encrypted integration credentials
#   attachment_data   the uploaded file bytes those rows point at
#   qdrant_data       the retrieval vectors
#
# The other four are re-derivable and are deliberately skipped. `ollama_models`
# and `infinity_cache` are downloaded weights - a re-pull replaces them exactly,
# and on a deployment whose model server is a separate machine the volume is not
# even here. `minio_data` only exists under an opt-in profile. `redis_data` is
# the background-job queue and nothing else: losing it loses queued work, not
# records, and the worker's stall sweep eventually fails any upload row whose job
# vanished, so the loss is visible and bounded rather than silent.
#
# WHY POSTGRES IS DUMPED RATHER THAN COPIED. Tarring a live postgres volume
# produces a file that LOOKS fine and is not: the data directory is written
# continuously, so a copy taken over several seconds captures pages from
# different moments, which is torn state a restore may or may not survive.
# `pg_dump` asks the running server for a snapshot instead - one transaction,
# one consistent view, no need to stop anything. The custom format (`-Fc`) is
# used because it is compressed and because `pg_restore` can list its contents,
# which is what the verification step below reads.
#
# WHY QDRANT IS SNAPSHOTTED RATHER THAN COPIED. Same reason, same shape of
# answer: Qdrant has its own snapshot API that flushes a collection to a single
# consistent archive. Two details worth knowing, both established against a
# running v1.18.2 instance:
#
#   - the archive lands in /qdrant/snapshots, which is NOT inside the
#     qdrant_data volume - it sits on the container's own writable layer, so it
#     survives no container recreation and consumes the Docker data root rather
#     than the volume. This script therefore downloads each snapshot over the
#     HTTP API and deletes the server-side copy immediately.
#   - the `checksum` the API reports is the SHA256 of exactly those downloaded
#     bytes, so the download can be verified against the source without trusting
#     the transfer.
#
# THAT HTTP API IS REACHED FROM INSIDE THE STACK'S NETWORK, never from a
# published host port. The production overlay publishes no store ports at all,
# and production is exactly where a backup matters - so a host-side client
# would work in development and be useless where it counts. Every request here
# runs in a throwaway pinned-curl container joined to the stack's own network,
# addressing Qdrant by service name: one code path, identical in both layouts,
# and no dependence on BIND_HOST or QDRANT_HTTP_PORT.
#
# ATTACHMENTS DEPEND ON WHICH STORAGE BACKEND IS CONFIGURED, and this is the
# one thing here that can be silently incomplete. With STORAGE_BACKEND=local
# the bytes are files in a volume and a tar is the whole story. With `s3` they
# are objects in a bucket, that volume is empty or stale, and tarring it would
# yield an archive that looks like the uploaded files and holds none of them -
# so the step is SKIPPED, loudly, and the manifest records that it was. Backing
# up a bucket needs the provider's tooling and credentials this script is not
# given; pretending otherwise is worse than saying so.
#
# Under the local backend they are plain files, so a tar is the whole story.
# It is taken through a throwaway container with the volume mounted READ-ONLY,
# which touches no running service. Backend and worker write those files with a
# temp-file-then-rename, so an in-flight upload is either fully present or
# absent from the tar - never half-written.
#
# Usage:
#   sh scripts/backup.sh [DEST]        back up into DEST/<timestamp>/
#   sh scripts/backup.sh plan [DEST]   print what it would do; change nothing
#   sh scripts/backup.sh safe-name <n> answer ok/unsafe for one artifact name
#   sh scripts/backup.sh collection-names   parse a /collections reply on stdin
#   sh scripts/backup.sh network-name       the stack's Docker network
#   sh scripts/backup.sh storage-captures-attachments <backend>   yes/no
#
# DEST defaults to ./backups (git-ignored). Each run creates its own
# timestamped subdirectory, so pointing DEST at a backup disk builds a series
# rather than overwriting one copy. Note that an unrecognised first argument is
# TAKEN as DEST, so a mistyped subcommand writes a directory of that name; a
# real run prints where it is writing before it writes anything.
#
# The last three forms read their input rather than the stack, so the guard,
# the parser and the network rule can each be exercised without Docker, a
# stack, or a backup to hand.

set -eu

ROOT=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
ENV_FILE="$ROOT/.env"

# --- pure helpers (no Docker, no network) ------------------------------------

# One key's value out of the env file, matching how the Makefile and
# scripts/remote_check.sh read it: the LAST uncommented assignment wins,
# carriage returns are stripped (this file is edited on Windows as often as
# not), and surrounding quotes belong to the file format rather than the value.
env_value() {
	[ -f "$ENV_FILE" ] || return 0
	value=$(tr -d '\r' < "$ENV_FILE" | sed -n "s/^[[:space:]]*$1=//p" | tail -n 1)
	case "$value" in
	\"*\")
		value=${value#\"}
		value=${value%\"}
		;;
	\'*\')
		value=${value#\'}
		value=${value%\'}
		;;
	esac
	printf '%s' "$value"
}

env_or() {
	v=$(env_value "$1")
	if [ -n "$v" ]; then printf '%s' "$v"; else printf '%s' "$2"; fi
}

# Is this name safe to use as a path component?
#
# Collection names arrive from Qdrant's API, and every one of them becomes a
# file name here and a path inside the Qdrant container on the way back. A name
# containing a slash or a `..` segment would place a file outside the backup
# directory - or, on restore, outside the snapshots directory in the container.
# A name starting with `-` is read as an option by whatever command receives
# it, and a quote or a space breaks out of the quoting that carries it.
#
# Deliberately an allowlist: letters, digits, dot, dash, underscore. Anything
# else is refused rather than escaped, because "escape it correctly everywhere"
# is a promise a shell script cannot keep.
safe_name() {
	n=${1:-}
	if [ -z "$n" ]; then
		echo "unsafe"
		return 0
	fi
	case "$n" in
	-* | *..* | */* | *\\*)
		echo "unsafe"
		return 0
		;;
	esac
	case "$n" in
	*[!A-Za-z0-9._-]*)
		echo "unsafe"
		return 0
		;;
	esac
	echo "ok"
}

require_safe_name() {
	if [ "$(safe_name "${1:-}")" != "ok" ]; then
		echo "backup: refusing to use '${1:-}' as a file name." >&2
		echo "  Only letters, digits, dot, dash and underscore are accepted - anything" >&2
		echo "  else could write outside the backup directory." >&2
		exit 1
	fi
}

# The compose project name, which prefixes every named volume and container.
# Read from the compose file rather than hardcoded, and overridden by the same
# environment variable Docker itself honours.
project_name() {
	if [ -n "${COMPOSE_PROJECT_NAME:-}" ]; then
		printf '%s' "$COMPOSE_PROJECT_NAME"
		return 0
	fi
	sed -n 's/^name:[[:space:]]*//p' "$ROOT/infra/docker-compose.yml" | head -n 1 | tr -d '\r'
}

# The Docker network the stack's services share, as DECLARED by the compose
# file, falling back to Compose's own default naming.
#
# Not a detail: Compose names a project's default network `<project>_default`
# UNLESS the file renames it, and this one does - `networks: default: name:` is
# set near the bottom of infra/docker-compose.yml. Code that assumes the
# `<project>_default` form finds no such network here and reports the stack as
# unreachable while it is running perfectly.
#
# This is the fallback answer. The authoritative one is read off the running
# Qdrant container (see qdrant_network below), because that is the network the
# helper has to join to reach it. Pure, so the rule can be checked without a
# daemon.
# Does this storage backend keep attachment bytes in the local volume?
#
# The attachment tar is a copy of the `attachment_data` volume, which is where
# the LOCAL adapter writes. With STORAGE_BACKEND=s3 the bytes are in a bucket
# instead and that volume is empty or stale - so tarring it would produce an
# artifact that looks like the uploaded files, restores cleanly, and contains
# nothing anyone uploaded. Skipping loudly is the only honest answer: this
# script cannot back up a bucket (that needs the provider's own tooling and
# credentials it is not given).
#
# Anything that is not exactly "local" counts as NOT captured. The backend
# accepts only local|s3 and fails at boot on anything else, so an unrecognised
# value here means a misconfigured deployment - and the safe reading of a
# misconfiguration is "assume the bytes are not in that volume", never the
# reverse.
storage_captures_attachments() {
	case "${1:-}" in
	local) echo "yes" ;;
	*) echo "no" ;;
	esac
}

compose_network_name() {
	declared=$(awk '
		/^networks:/ { in_net = 1; next }
		/^[^[:space:]]/ { in_net = 0 }
		in_net && /^[[:space:]]+name:[[:space:]]*/ {
			sub(/^[[:space:]]+name:[[:space:]]*/, "")
			print
			exit
		}
	' "$ROOT/infra/docker-compose.yml" | tr -d '\r')
	if [ -n "$declared" ]; then
		printf '%s' "$declared"
	else
		printf '%s_default' "$(project_name)"
	fi
}

# The file is fed on STDIN rather than named as an argument. Given a name, GNU
# coreutils escapes any backslash or newline in it AND prefixes the whole
# output line with a backslash - so a destination path spelled the Windows way
# yields "\<hash>", which then matches nothing and reports a corrupt artifact
# while printing what looks like the right hash.
sha256_of() {
	if command -v sha256sum >/dev/null 2>&1; then
		sha256sum < "$1" | cut -d' ' -f1
	elif command -v shasum >/dev/null 2>&1; then
		shasum -a 256 < "$1" | cut -d' ' -f1
	else
		echo "backup: no sha256sum or shasum on PATH - cannot checksum artifacts." >&2
		exit 1
	fi
}

file_size() {
	# `wc -c` rather than `stat`, whose flags differ between GNU and BSD.
	wc -c < "$1" | tr -d ' \t'
}

# Pull one string field out of a small JSON object. Enough for Qdrant's
# snapshot replies, which are flat and machine-generated; not a JSON parser and
# not used on anything richer.
json_field() {
	printf '%s' "$1" | sed -n "s/.*\"$2\":\"\([^\"]*\)\".*/\1/p" | head -n 1
}

# --- the stack ---------------------------------------------------------------

PROJECT=$(project_name)
PG_USER=$(env_or POSTGRES_USER chatbot)
PG_DB=$(env_or POSTGRES_DB chatbot)
STORAGE_BACKEND=$(env_or STORAGE_BACKEND local)
ATTACHMENTS_CAPTURED=$(storage_captures_attachments "$STORAGE_BACKEND")
# Qdrant is addressed by SERVICE NAME on the stack's own network, never through
# a published host port. See qdrant_curl below for why that is the only path.
# 6333 is the container's port, fixed in the compose file; QDRANT_HTTP_PORT
# only renames the HOST side and is irrelevant here.
QDRANT_URL="http://qdrant:6333"
ATTACHMENT_VOLUME="${PROJECT}_attachment_data"
# Only ever used to read a volume and to unpack a tar; the stack already pulls
# far larger images, and this one is ~5 MB.
HELPER_IMAGE=alpine
# Pinned: a backup that reaches for a moving `latest` can change behaviour
# between two runs of the same script. ~35 MB, pulled once.
CURL_IMAGE=curlimages/curl:8.21.0

# Git Bash rewrites any argument that looks like a Unix path into a Windows one
# before a Windows-side program sees it, which turns `/src` into
# `C:/Program Files/Git/src` and makes every container path wrong. Harmless
# everywhere else - it is just an unread variable.
docker_() {
	MSYS_NO_PATHCONV=1 docker "$@"
}

# The container running one compose service, found by the labels Compose sets
# rather than by assembling a name - a name depends on the replica suffix, the
# labels do not.
service_container() {
	docker_ ps -q \
		--filter "label=com.docker.compose.project=$PROJECT" \
		--filter "label=com.docker.compose.service=$1" | head -n 1
}

need_container() {
	cid=$(service_container "$1")
	if [ -z "$cid" ]; then
		echo "backup: the '$1' container is not running." >&2
		echo "  A backup reads the live services, so the stack has to be up: 'make up'." >&2
		exit 1
	fi
	printf '%s' "$cid"
}

describe_container() {
	cid=$(service_container "$1")
	if [ -n "$cid" ]; then printf '%s' "$cid"; else printf '%s' "(not running)"; fi
}

# The network the running Qdrant container is actually attached to.
#
# Read from the container rather than composed from a naming convention: the
# convention is wrong here (the compose file renames the default network), and
# a wrong answer means the helper below joins a network Qdrant is not on and
# fails to resolve the service name.
qdrant_network() {
	cid=$(service_container qdrant)
	if [ -n "$cid" ]; then
		net=$(docker_ inspect \
			-f '{{range $k,$v := .NetworkSettings.Networks}}{{$k}}{{"\n"}}{{end}}' "$cid" \
			| tr -d '\r' | sed '/^$/d' | head -n 1)
		if [ -n "$net" ]; then
			printf '%s' "$net"
			return 0
		fi
	fi
	compose_network_name
}

# EVERY Qdrant request goes through here, and that is the point.
#
# The obvious alternative - curl from the host to a published port - works in
# development and fails exactly where a backup matters most: the production
# overlay publishes NO store ports (`ports: !reset []`), so a host-side client
# cannot reach Qdrant at all there. Rather than carry two paths and test one,
# this runs curl in a throwaway container attached to the stack's own network
# and addresses Qdrant by service name, which is identical in both layouts.
#
# It also means the host needs no curl, and that the address does not depend on
# BIND_HOST, QDRANT_HTTP_PORT, or which interface the port was published on.
qdrant_curl() {
	docker_ run --rm --network "$QDRANT_NETWORK" "$CURL_IMAGE" "$@"
}

# One collection name per LINE, read out of a /collections response body.
#
# Line-per-name rather than a space-separated list so a name containing a space
# arrives whole and is rejected by safe_name with a message that names it,
# instead of silently becoming two names that each 404.
#
# Pure - the body is an argument - so the parser that decides which collections
# get backed up can be exercised against a fixture. That decision is the one
# place where a quiet bug costs the whole vector store: a parser that finds
# nothing produces a backup that reports success and contains no vectors.
collection_names() {
	printf '%s' "${1:-}" | tr ',' '\n' | sed -n 's/.*"name":"\([^"]*\)".*/\1/p'
}

# The raw /collections body, or a stopped script.
#
# The status is checked explicitly rather than piped away: piping curl into
# `sed` hands the exit status to `sed`, which always succeeds, so a Qdrant that
# stopped answering between the preflight and here would yield an EMPTY list -
# reported as "no collections", written into the manifest as a complete backup,
# and discovered during a restore that puts back no vectors at all.
fetch_collections() {
	body=$(qdrant_curl -fsS "$QDRANT_URL/collections") || {
		echo "backup: Qdrant stopped answering at $QDRANT_URL while listing collections." >&2
		echo "  Refusing to finish: an empty list here is indistinguishable from a vector" >&2
		echo "  store that really is empty, and the backup would claim to be complete." >&2
		exit 1
	}
	printf '%s' "$body"
}

# --- preflight ---------------------------------------------------------------

preflight() {
	# No curl here on purpose: every HTTP call this script makes runs inside a
	# container, so the host needs Docker and tar and nothing else.
	for tool in docker tar; do
		command -v "$tool" >/dev/null 2>&1 || {
			echo "backup: '$tool' is not on PATH, and the backup needs it." >&2
			exit 1
		}
	done

	if ! docker_ info >/dev/null 2>&1; then
		echo "backup: cannot reach the Docker daemon - is Docker running?" >&2
		exit 1
	fi

	# Only when the attachment bytes are actually IN that volume. Under
	# STORAGE_BACKEND=s3 it may not even exist, and its absence is then correct
	# rather than a fault to refuse over.
	if [ "$ATTACHMENTS_CAPTURED" = "yes" ] &&
		! docker_ volume inspect "$ATTACHMENT_VOLUME" >/dev/null 2>&1; then
		echo "backup: no volume named '$ATTACHMENT_VOLUME'." >&2
		echo "  Volumes are prefixed with the compose project name. This run resolved the" >&2
		echo "  project as '$PROJECT'; set COMPOSE_PROJECT_NAME if the stack uses another." >&2
		exit 1
	fi

	QDRANT_NETWORK=$(qdrant_network)
	if ! docker_ network inspect "$QDRANT_NETWORK" >/dev/null 2>&1; then
		echo "backup: no Docker network named '$QDRANT_NETWORK'." >&2
		echo "  Qdrant is reached from a helper container joined to the stack's own" >&2
		echo "  network, so that network has to exist. Start the stack ('make up')." >&2
		exit 1
	fi

	# One reachability check, through exactly the mechanism every later call
	# uses - so a helper image that cannot be pulled, or a network the service
	# name does not resolve on, is found here rather than half way through.
	if ! qdrant_curl -fsS -o /dev/null "$QDRANT_URL/collections"; then
		echo "backup: Qdrant does not answer at $QDRANT_URL from network '$QDRANT_NETWORK'." >&2
		echo "  The snapshot API is the only consistent way to copy the vectors. This" >&2
		echo "  reaches it from a throwaway $CURL_IMAGE container on the stack's own" >&2
		echo "  network, so no published port is involved - the likely causes are the" >&2
		echo "  stack being down, or that image being unavailable to pull." >&2
		exit 1
	fi
}

# --- plan --------------------------------------------------------------------

# Prints the resolved inputs and the artifacts a real run would write. Reads
# the stack (which containers, which collections) and writes nothing, so it is
# safe at any time and is the cheapest way to check a deployment is wired the
# way this script expects.
plan() {
	preflight
	echo "== backup plan =="
	echo "  destination      $DEST_ROOT/<timestamp>"
	echo "  compose project  $PROJECT"
	echo "  postgres         $(describe_container postgres) (database '$PG_DB', user '$PG_USER')"
	echo "  qdrant           $(describe_container qdrant) at $QDRANT_URL"
	echo "                   reached from network '$QDRANT_NETWORK' via $CURL_IMAGE"
	if [ "$ATTACHMENTS_CAPTURED" = "yes" ]; then
		echo "  attachments      volume $ATTACHMENT_VOLUME (read-only mount, image $HELPER_IMAGE)"
	else
		echo "  attachments      NOT COVERED - STORAGE_BACKEND=$STORAGE_BACKEND"
	fi
	echo ""
	echo "  Would write:"
	echo "    postgres.dump          pg_dump -Fc of '$PG_DB'"
	collection_names "$(fetch_collections)" | while IFS= read -r coll; do
		[ -n "$coll" ] || continue
		echo "    qdrant/$coll.snapshot  collection snapshot (created, downloaded, deleted server-side)"
	done
	if [ "$ATTACHMENTS_CAPTURED" = "yes" ]; then
		echo "    attachments.tar        tar of $ATTACHMENT_VOLUME"
	else
		echo "    (no attachments.tar)   uploaded bytes are in object storage, not that volume"
	fi
	echo "    manifest.txt           timestamp, git revision, alembic head, sizes and SHA256 of each"
	echo ""
	echo "  Not copied, by design: redis_data (job queue only), ollama_models and"
	echo "  infinity_cache (re-pullable weights), minio_data (opt-in profile)."
	echo "  See docs/DATA-MODEL.md, 'What lives in each volume'."
}

# --- run ---------------------------------------------------------------------

run() {
	preflight

	# Both containers are resolved BEFORE the destination directory is created,
	# so a run against a half-started stack fails without leaving an empty
	# timestamped directory behind that reads as a backup in `ls backups`.
	PG_CID=$(need_container postgres)
	QDRANT_CID=$(need_container qdrant)

	STAMP=$(date -u +%Y%m%d-%H%M%S)
	OUT="$DEST_ROOT/$STAMP"
	if [ -e "$OUT" ]; then
		echo "backup: $OUT already exists - refusing to write into it." >&2
		exit 1
	fi
	mkdir -p "$OUT/qdrant"
	# Announced before the first byte is written. An unrecognised first argument
	# is taken as the destination, so a mistyped subcommand becomes a directory
	# rather than an error - saying where this is going makes that visible on
	# the first line instead of being found later in an `ls`.
	echo "Writing to $OUT"

	MANIFEST="$OUT/manifest.txt"
	{
		echo "# AI Chatbot Platform backup manifest"
		echo "# Restore with: make restore DIR=$OUT"
		echo "created_utc: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
		echo "compose_project: $PROJECT"
		echo "postgres_db: $PG_DB"
		echo "postgres_user: $PG_USER"
		# What this backup DOES NOT contain is as load-bearing as what it does.
		# The restore reads `attachments_captured` and refuses to touch the
		# attachment volume when it says no - otherwise it would wipe a volume
		# and unpack nothing over it, which on a mis-set STORAGE_BACKEND is how
		# a restore destroys the files it was run to protect.
		echo "storage_backend: $STORAGE_BACKEND"
		echo "attachments_captured: $ATTACHMENTS_CAPTURED"
		# Both recorded because an archive is only readable FORWARDS: a dump
		# taken on a newer server cannot be restored by an older one, and a
		# snapshot is tied to the Qdrant that wrote it. The restore re-checks
		# the postgres half against the live server before it drops anything;
		# these lines are what tells a person which version to bring back.
		echo "postgres_server_version: $(docker_ exec "$PG_CID" \
			psql -U "$PG_USER" -d "$PG_DB" -tAc 'SHOW server_version' 2>/dev/null \
			| tr -d '\r' | head -n 1 || echo unknown)"
		echo "qdrant_container_image: $(docker_ inspect -f '{{.Config.Image}}' "$QDRANT_CID")"
	} > "$MANIFEST"

	# The commit the source tree was on. Recorded because it is the only thing
	# that explains an alembic head later, and absent from a deployment that
	# unpacked a release rather than cloning - which is a fact, not an error.
	if git -C "$ROOT" rev-parse HEAD >/dev/null 2>&1; then
		echo "git_rev: $(git -C "$ROOT" rev-parse HEAD)" >> "$MANIFEST"
	else
		echo "git_rev: unknown (not a git checkout)" >> "$MANIFEST"
	fi

	# THE FIELD THE RESTORE CARES ABOUT MOST. The backend migrates on boot, so a
	# dump from OLDER code heals itself; a dump from NEWER code cannot, and this
	# is what lets the restore say which of the two it is holding.
	HEAD=$(docker_ exec "$PG_CID" psql -U "$PG_USER" -d "$PG_DB" -tAc \
		"SELECT version_num FROM alembic_version" 2>/dev/null | tr -d '\r' | head -n 1 || true)
	if [ -z "$HEAD" ]; then
		HEAD="unknown"
		echo "backup: WARNING - could not read alembic_version; the restore will not be able" >&2
		echo "  to tell whether this dump is ahead of the code it is restored into." >&2
	fi
	echo "alembic_head: $HEAD" >> "$MANIFEST"

	echo "== 1/4 postgres =="
	# No TTY is allocated, so stdout is byte-exact and the custom-format dump
	# survives the trip through the shell unchanged.
	docker_ exec "$PG_CID" pg_dump -U "$PG_USER" -d "$PG_DB" -Fc > "$OUT/postgres.dump"
	echo "  postgres.dump  $(file_size "$OUT/postgres.dump") bytes"

	echo "== 2/4 qdrant =="
	# Via a file, so the loop body runs in THIS shell (a pipeline's body runs in
	# a subshell, where COLL_LIST would be discarded) while still reading one
	# whole name per line.
	COLL_FILE="$OUT/.collections"
	COLL_BODY=$(fetch_collections)
	collection_names "$COLL_BODY" > "$COLL_FILE"
	if [ ! -s "$COLL_FILE" ]; then
		echo "  no collections - nothing to snapshot (a stack that has never ingested)"
	fi
	COLL_LIST=""
	while IFS= read -r coll; do
		[ -n "$coll" ] || continue
		require_safe_name "$coll"
		reply=$(qdrant_curl -fsS -X POST "$QDRANT_URL/collections/$coll/snapshots?wait=true")
		snap=$(json_field "$reply" name)
		want=$(json_field "$reply" checksum)
		if [ -z "$snap" ]; then
			echo "backup: Qdrant did not name a snapshot for '$coll'. Reply: $reply" >&2
			exit 1
		fi

		# Streamed out through the helper's stdout into a host file. The
		# redirect is the shell's, so the bytes never become a command-line
		# argument and no path translation applies to them.
		qdrant_curl -fsS "$QDRANT_URL/collections/$coll/snapshots/$snap" \
			> "$OUT/qdrant/$coll.snapshot"

		# Delete the server-side copy whatever happens next: it lives on the
		# container's writable layer, outside any volume, so leaving one behind
		# quietly consumes the Docker data root until the container is recreated.
		qdrant_curl -fsS -o /dev/null -X DELETE \
			"$QDRANT_URL/collections/$coll/snapshots/$snap?wait=true" || {
			echo "backup: WARNING - could not delete the server-side snapshot '$snap'." >&2
			echo "  Remove it by hand; it is inside the qdrant container at" >&2
			echo "  /qdrant/snapshots/$coll/, which is not a volume." >&2
		}

		# The API reports the SHA256 of the snapshot it wrote, so this compares
		# the bytes on disk against the source rather than against themselves.
		got=$(sha256_of "$OUT/qdrant/$coll.snapshot")
		if [ -n "$want" ] && [ "$got" != "$want" ]; then
			echo "backup: '$coll.snapshot' does not match the checksum Qdrant reported." >&2
			echo "  expected $want" >&2
			echo "  got      $got" >&2
			exit 1
		fi
		echo "  qdrant/$coll.snapshot  $(file_size "$OUT/qdrant/$coll.snapshot") bytes"
		COLL_LIST="$COLL_LIST $coll"
	done < "$COLL_FILE"
	rm -f "$COLL_FILE"
	echo "qdrant_collections:$COLL_LIST" >> "$MANIFEST"

	echo "== 3/4 attachments =="
	if [ "$ATTACHMENTS_CAPTURED" = "yes" ]; then
		# Read-only mount, throwaway container: nothing that is running is
		# touched, and the volume cannot be modified even if the tar misbehaves.
		docker_ run --rm -v "$ATTACHMENT_VOLUME:/src:ro" "$HELPER_IMAGE" \
			tar -cf - -C /src . > "$OUT/attachments.tar"
		echo "  attachments.tar  $(file_size "$OUT/attachments.tar") bytes"
	else
		# Said at length, and repeated in the closing summary, because it is the
		# one gap in an otherwise complete backup - and a gap nobody reads about
		# is a gap discovered during a restore.
		echo "  SKIPPED - STORAGE_BACKEND=$STORAGE_BACKEND"
		echo ""
		echo "  !! UPLOADED FILES ARE NOT IN THIS BACKUP."
		echo ""
		echo "     With this backend the attachment bytes live in object storage, not"
		echo "     in the $ATTACHMENT_VOLUME volume - so there is nothing here to"
		echo "     copy, and tarring that volume would produce an archive that looks"
		echo "     like the uploaded files and contains none of them."
		echo ""
		echo "     Back the BUCKET up with the provider's own tooling (versioning,"
		echo "     replication, or a sync to another bucket). This script has neither"
		echo "     the credentials nor a safe way to do it, and does not try."
		echo ""
		echo "     The database rows that point at those objects ARE backed up, so a"
		echo "     restore without the bucket leaves every attachment row pointing at"
		echo "     bytes that are not there."
		echo ""
	fi

	echo "== 4/4 verify =="
	# Each artifact is checked with the tool that would READ it during a
	# restore, not merely for being non-empty. A backup nobody can open is
	# indistinguishable from no backup, and it stays that way until the worst
	# possible moment.
	docker_ exec -i "$PG_CID" pg_restore --list > /dev/null < "$OUT/postgres.dump" || {
		echo "backup: pg_restore could not read postgres.dump - the dump is not usable." >&2
		exit 1
	}
	echo "  postgres.dump    readable by pg_restore --list"

	# Fed on stdin rather than by name. GNU tar reads a leading `host:` in a
	# filename as a REMOTE archive, so a destination on a Windows drive
	# (`C:/backups/...`) makes it try to connect to a machine called `C` -
	# which fails with a message about resolving a host, not about the archive.
	if [ "$ATTACHMENTS_CAPTURED" = "yes" ]; then
		tar -tf - < "$OUT/attachments.tar" > /dev/null || {
			echo "backup: attachments.tar is not a readable archive." >&2
			exit 1
		}
		echo "  attachments.tar  readable by tar -t"
	else
		echo "  attachments      not captured (STORAGE_BACKEND=$STORAGE_BACKEND)"
	fi

	for coll in $COLL_LIST; do
		if [ ! -s "$OUT/qdrant/$coll.snapshot" ]; then
			echo "backup: qdrant/$coll.snapshot is empty." >&2
			exit 1
		fi
	done
	echo "  qdrant snapshots non-empty and checksum-matched at download"

	# Written last so a manifest exists only for artifacts that all verified.
	echo "" >> "$MANIFEST"
	echo "# artifact <path> <bytes> <sha256>" >> "$MANIFEST"
	(
		cd "$OUT"
		for f in postgres.dump attachments.tar qdrant/*.snapshot; do
			[ -f "$f" ] || continue
			echo "artifact $f $(file_size "$f") $(sha256_of "$f")"
		done
	) >> "$MANIFEST"

	echo ""
	echo "Backup complete: $OUT"
	echo "  alembic head $HEAD - restoring this into code NEWER than the dump is fine"
	echo "  (migrations run forward on boot); restoring it into OLDER code is not."
	if [ "$ATTACHMENTS_CAPTURED" != "yes" ]; then
		echo ""
		echo "  INCOMPLETE: uploaded files are NOT in this backup (STORAGE_BACKEND="
		echo "  $STORAGE_BACKEND). Back up the object-storage bucket separately with the"
		echo "  provider's tooling; the database rows referencing it are covered here."
	fi
	echo "  Restore with: make restore DIR=$OUT"
}

# --- dispatch ----------------------------------------------------------------
# An unrecognised first argument is treated as DEST rather than rejected, so the
# common form stays `backup.sh /mnt/backups`. The two subcommands are ordinary
# words that no plausible destination path is spelled as.

case "${1:-}" in
safe-name)
	safe_name "${2:-}"
	;;
collection-names)
	# The body arrives on STDIN, not as an argument: a JSON document is mostly
	# quote characters, and quotes do not survive being passed through every
	# host's command line intact.
	collection_names "$(cat)"
	;;
network-name)
	compose_network_name
	;;
storage-captures-attachments)
	storage_captures_attachments "${2:-}"
	;;
plan)
	DEST_ROOT=${2:-$ROOT/backups}
	plan
	;;
*)
	DEST_ROOT=${1:-$ROOT/backups}
	run
	;;
esac
