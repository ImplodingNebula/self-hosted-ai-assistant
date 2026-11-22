#!/bin/sh
# Put a backup taken by scripts/backup.sh back. Invoked by `make restore DIR=...`.
#
# WHAT THIS REFUSES TO DO, and why each refusal is worth the friction:
#
#   - it will not run while the application services are up. The backend and
#     the worker hold open database connections (so the database cannot be
#     dropped), they write attachment files (so a tar extracted underneath them
#     races), and a chat turn in flight would write rows into a database that
#     is about to be replaced. The stores - postgres, qdrant - stay UP on
#     purpose: this restores THROUGH them rather than around them.
#   - it will not start or stop anything itself. Deciding when an application
#     goes down is the operator's call, and a script that stops services as a
#     side effect of a restore is a script nobody can run to look at a plan.
#     The only containers it creates are throwaway `docker run --rm` helpers
#     that mount a volume and exit.
#   - it will not restore artifacts whose checksums do not match the manifest.
#
# Qdrant is reached from a throwaway curl container joined to the stack's own
# network, addressing it by service name - never through a published host port,
# because the production overlay publishes none and production is where a
# restore matters. One code path, identical in development and deployment.
#
# THE SHARP EDGE: A DUMP FROM CODE NEWER THAN THIS CHECKOUT.
# The backend image runs `alembic upgrade head` before it serves, so the
# database schema catches up to the code automatically at boot. That makes one
# direction free and the other fatal:
#
#   dump OLDER than the code   the pending migrations run on the next boot and
#                              the restore heals itself. Nothing to do.
#   dump NEWER than the code   the database is at a revision this checkout has
#                              never heard of. Alembic cannot walk from a
#                              revision it does not have, so the backend exits
#                              at boot with "Can't locate revision identified
#                              by <id>" and the stack never comes up. Nothing
#                              is corrupted - the fix is to deploy code at or
#                              after the commit recorded in the manifest - but
#                              the failure arrives minutes later, wearing a
#                              message that says nothing about a restore.
#
# So the head recorded in the manifest is compared against the migrations in
# this checkout before anything is written, and the verdict is printed at the
# start and repeated at the end.
#
# Usage:
#   sh scripts/restore.sh <DIR>          restore the backup in DIR
#   sh scripts/restore.sh plan <DIR>     verify DIR, print the plan, change nothing
#   sh scripts/restore.sh verify <DIR>   check DIR only - no Docker, no stack
#   sh scripts/restore.sh collections <DIR>  list the collections it would recover
#   sh scripts/restore.sh classify-running <project> [name...]
#   sh scripts/restore.sh classify-head <head> [versions-dir]
#   sh scripts/restore.sh safe-name <n>  answer ok/unsafe for one collection name
#   sh scripts/restore.sh collection-names  parse a /collections reply on stdin
#   sh scripts/restore.sh network-name      the stack's Docker network
#   sh scripts/restore.sh attachments-plan <DIR>   would step 5 restore or skip
#   sh scripts/restore.sh storage-captures-attachments <backend>  yes/no
#   sh scripts/restore.sh compose-files     the -f pair for this DEPLOY_ENV
#
# Every form below the first two takes its input as an argument or a directory
# to read, so each refusal rule can be exercised without a stack, a Docker
# daemon, or anything being restored.

set -eu

ROOT=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
ENV_FILE="$ROOT/.env"

# Services that must be DOWN before a restore. The stores are deliberately not
# in this list - they are what the restore talks to.
APP_SERVICES="backend worker frontend"

# --- pure helpers (no Docker, no network) ------------------------------------

# See scripts/remote_check.sh for the reasoning; same format, same rules.
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

project_name() {
	if [ -n "${COMPOSE_PROJECT_NAME:-}" ]; then
		printf '%s' "$COMPOSE_PROJECT_NAME"
		return 0
	fi
	sed -n 's/^name:[[:space:]]*//p' "$ROOT/infra/docker-compose.yml" | head -n 1 | tr -d '\r'
}

# The `-f` pair that names this machine's compose layout, mirroring the rule in
# the Makefile.
#
# This script never RUNS compose - it only prints commands for a person to run.
# Printing the wrong pair is still a real hazard rather than a cosmetic one: the
# development overlay bind-mounts source and publishes ports, so pasting it on a
# deployment recreates those containers from the development layout, which is
# precisely what DEPLOY_ENV exists to stop happening by accident.
compose_files() {
	case "$(env_value DEPLOY_ENV)" in
	prod) printf -- '-f infra/docker-compose.yml -f infra/docker-compose.prod.yml' ;;
	*) printf -- '-f infra/docker-compose.yml -f infra/docker-compose.override.yml' ;;
	esac
}

# Does this storage backend keep attachment bytes in the local volume?
# The twin of the same helper in scripts/backup.sh; a test asserts they agree.
storage_captures_attachments() {
	case "${1:-}" in
	local) echo "yes" ;;
	*) echo "no" ;;
	esac
}

# The Docker network the stack's services share, as DECLARED by the compose
# file, falling back to Compose's own default naming.
#
# The twin of the same helper in scripts/backup.sh, and the reason it is not
# just `<project>_default`: this compose file renames its default network, so
# the conventional form names a network that does not exist. Pure, so the rule
# can be checked without a daemon.
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

# Which of the given container names belong to an application service of this
# project. Prints the offenders, one per line, and nothing at all when the
# coast is clear - so an empty answer is the go-ahead.
#
# Matching is on Compose's own naming (<project>-<service>-<replica>) rather
# than on a substring, because a substring test says yes to another project's
# `myapp-backend-1` and blocks a restore for a container that has nothing to do
# with this stack.
classify_running() {
	proj=${1:-}
	shift 2>/dev/null || true
	for name in "$@"; do
		for svc in $APP_SERVICES; do
			case "$name" in
			"$proj-$svc-"*)
				# Replica suffix must be digits: `proj-backend-1` counts,
				# `proj-backend-sidecar-1` is a different service.
				suffix=${name#"$proj-$svc-"}
				case "$suffix" in
				'' | *[!0-9]*) ;;
				*) echo "$name" ;;
				esac
				;;
			esac
		done
	done
}

# Is this name safe to use as a path component and in a URL?
#
# The same allowlist scripts/backup.sh applies on the way out, applied again on
# the way IN - because a restore reads its collection names from FILENAMES in a
# directory, and a backup directory is a thing that gets copied between
# machines, renamed by hand, and restored from a shared drive. A name with a
# space or a quote in it would break out of the quoting in the copy-in command
# below; the two scripts are not allowed to disagree about what is acceptable.
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

# One collection name per line, read out of a /collections response body.
#
# The twin of the same helper in scripts/backup.sh. Duplicated rather than
# shared because these are two standalone scripts an operator may copy one of;
# a test asserts the two agree, so "change one, change both" is a failure
# rather than a comment.
collection_names() {
	printf '%s' "${1:-}" | tr ',' '\n' | sed -n 's/.*"name":"\([^"]*\)".*/\1/p'
}

# Does this checkout contain the migration the dump ends at?
#
#   known      the revision is one of ours - the restore heals forward on boot
#   unknown    the dump is ahead of (or beside) this code - the backend will
#              refuse to start until newer code is deployed
#   unchecked  there are no migrations here to compare against, so no verdict
#              is possible. Deliberately its own answer: reporting "unknown"
#              for a missing directory would raise a false alarm, and reporting
#              "known" would hide a real one.
classify_head() {
	head=${1:-}
	dir=${2:-$ROOT/apps/backend/alembic/versions}

	if [ -z "$head" ] || [ "$head" = "unknown" ]; then
		echo "unchecked"
		return 0
	fi
	if [ ! -d "$dir" ]; then
		echo "unchecked"
		return 0
	fi
	if ! ls "$dir"/*.py >/dev/null 2>&1; then
		echo "unchecked"
		return 0
	fi
	# The revision id as Alembic declares it at the top of a migration:
	#   revision: str = "c0d1e2f3a4b5"
	if grep -qE "^revision(: *str)? *= *[\"']$head[\"']" "$dir"/*.py 2>/dev/null; then
		echo "known"
	else
		echo "unknown"
	fi
}

# Fed on STDIN, not named - see the same helper in scripts/backup.sh for why
# (GNU coreutils escapes a name containing a backslash and prefixes the line
# with one, which turns every comparison here into a false "corrupt artifact").
sha256_of() {
	if command -v sha256sum >/dev/null 2>&1; then
		sha256sum < "$1" | cut -d' ' -f1
	elif command -v shasum >/dev/null 2>&1; then
		shasum -a 256 < "$1" | cut -d' ' -f1
	else
		echo "restore: no sha256sum or shasum on PATH - cannot verify artifacts." >&2
		exit 1
	fi
}

file_size() { wc -c < "$1" | tr -d ' \t'; }

manifest_value() {
	sed -n "s/^$1: *//p" "$DIR/manifest.txt" | head -n 1 | tr -d '\r'
}

# Did the backup capture the attachment volume, according to its own manifest?
#
# The answer decides whether the final step runs at all, and the DEFAULT when
# the field is absent is "yes": a manifest written before this field existed
# describes a backup that always tarred the volume, so treating a missing field
# as "no" would silently stop restoring files that are right there.
#
# The distinction is not cosmetic. Step 5 EMPTIES the volume before unpacking,
# so running it against a backup that captured nothing deletes whatever is in
# that volume and writes nothing back - a restore that destroys the files it
# was run to protect. Hence a refusal rather than a no-op.
manifest_captures_attachments() {
	value=$(manifest_value attachments_captured)
	case "$value" in
	'') echo "yes" ;;
	yes) echo "yes" ;;
	*) echo "no" ;;
	esac
}

# --- the stack ---------------------------------------------------------------

PROJECT=$(project_name)
PG_USER=$(env_or POSTGRES_USER chatbot)
PG_DB=$(env_or POSTGRES_DB chatbot)
# This DEPLOYMENT's backend, which may differ from the one the backup was taken
# under. Whether the archive is unpacked is decided by the MANIFEST; this only
# decides whether to warn that unpacking it will not be read by anything.
STORAGE_BACKEND=$(env_or STORAGE_BACKEND local)
# Addressed by SERVICE NAME on the stack's own network, never through a
# published host port - the production overlay publishes none, and that is
# exactly where a restore has to work. 6333 is the container's port, fixed in
# the compose file; QDRANT_HTTP_PORT only renames the host side.
QDRANT_URL="http://qdrant:6333"
ATTACHMENT_VOLUME="${PROJECT}_attachment_data"
HELPER_IMAGE=alpine
# Pinned so two runs of the same script cannot behave differently. ~35 MB.
CURL_IMAGE=curlimages/curl:8.21.0

# Git Bash rewrites Unix-looking arguments into Windows paths before a
# Windows-side program sees them, which mangles every container-side path.
docker_() {
	MSYS_NO_PATHCONV=1 docker "$@"
}

service_container() {
	docker_ ps -q \
		--filter "label=com.docker.compose.project=$PROJECT" \
		--filter "label=com.docker.compose.service=$1" | head -n 1
}

# The network the running Qdrant container is actually attached to. Read from
# the container rather than composed from a naming convention, because the
# convention is wrong for this stack - see compose_network_name above.
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

# EVERY Qdrant request goes through here - see the same helper in
# scripts/backup.sh for the reasoning. The short version: the production
# overlay publishes no store ports, so a host-side client cannot reach Qdrant
# where a restore matters most. One path, identical in dev and prod.
qdrant_curl() {
	docker_ run --rm --network "$QDRANT_NETWORK" "$CURL_IMAGE" "$@"
}

# Same, with stdin attached for a request body. The recover call carries JSON,
# which is mostly quote characters; `-d @-` on stdin keeps them out of a
# command line that a native Docker client would re-parse.
qdrant_curl_stdin() {
	docker_ run --rm -i --network "$QDRANT_NETWORK" "$CURL_IMAGE" "$@"
}

need_container() {
	cid=$(service_container "$1")
	if [ -z "$cid" ]; then
		echo "restore: the '$1' container is not running." >&2
		echo "  A restore writes THROUGH the running stores. Start them without the app:" >&2
		echo "    docker compose --env-file .env $(compose_files) up -d postgres qdrant" >&2
		exit 1
	fi
	printf '%s' "$cid"
}

# The listing's exit status is read explicitly rather than piped away. Piping
# into `tr` hands the status to `tr`, which always succeeds - so a docker
# command that failed would produce an empty list, which reads as "no
# application container is running" and waves the restore through while the
# backend is still writing. An unreadable state is never permission to
# continue.
running_app_containers() {
	names=$(docker_ ps --format '{{.Names}}' \
		--filter "label=com.docker.compose.project=$PROJECT") || {
		echo "restore: could not ask Docker which containers are running." >&2
		echo "  Stopping rather than assuming the application is down." >&2
		exit 1
	}
	names=$(printf '%s' "$names" | tr -d '\r')
	# shellcheck disable=SC2086  # names are container names: no spaces possible
	[ -z "$names" ] || classify_running "$PROJECT" $names
}

# --- verification ------------------------------------------------------------

verify_backup() {
	[ -d "$DIR" ] || {
		echo "restore: no such directory: $DIR" >&2
		exit 1
	}
	[ -f "$DIR/manifest.txt" ] || {
		echo "restore: $DIR has no manifest.txt, so it is not a backup this can read." >&2
		echo "  Backups are the timestamped directories 'make backup' creates." >&2
		exit 1
	}

	# Every artifact is checked against the manifest BEFORE anything is
	# written, so a truncated copy is caught while the live data is still
	# there rather than half way through replacing it.
	#
	# A literal carriage return, for stripping one off the last field below.
	CR=$(printf '\r')

	checked=0
	while read -r kind path want_size want_sha; do
		[ "$kind" = "artifact" ] || continue
		# Only the LAST field can carry a trailing carriage return, and it is
		# the checksum - so a manifest that has been through a Windows editor
		# fails every comparison while PRINTING two identical hashes, which is
		# the least debuggable error message this script could produce.
		want_sha=${want_sha%"$CR"}
		if [ ! -f "$DIR/$path" ]; then
			echo "restore: $path is listed in the manifest but missing from $DIR." >&2
			exit 1
		fi
		got_size=$(file_size "$DIR/$path")
		if [ "$got_size" != "$want_size" ]; then
			echo "restore: $path is $got_size bytes; the manifest says $want_size." >&2
			exit 1
		fi
		got_sha=$(sha256_of "$DIR/$path")
		if [ "$got_sha" != "$want_sha" ]; then
			echo "restore: $path does not match its manifest checksum - the copy is damaged." >&2
			echo "  expected $want_sha" >&2
			echo "  got      $got_sha" >&2
			exit 1
		fi
		checked=$((checked + 1))
	done < "$DIR/manifest.txt"

	if [ "$checked" -eq 0 ]; then
		echo "restore: the manifest lists no artifacts. Nothing to restore from $DIR." >&2
		exit 1
	fi

	# The artifacts whose ABSENCE is destructive, checked by name.
	#
	# Everything above validates what the manifest happens to list; this asserts
	# what it must list. Without it a directory that lost `postgres.dump` in a
	# copy - or had it deleted to save space - verifies perfectly, and the
	# restore then drops the live database before discovering there is nothing
	# to put back. The vector snapshots are deliberately NOT required: a stack
	# that has never ingested legitimately has none.
	#
	# attachments.tar is required only when the manifest says the backup
	# captured it. Under STORAGE_BACKEND=s3 the uploaded bytes are in a bucket,
	# the backup skipped the volume on purpose, and demanding the archive would
	# make every such backup unrestorable.
	required_set="postgres.dump"
	if [ "$(manifest_captures_attachments)" = "yes" ]; then
		required_set="$required_set attachments.tar"
	fi
	for required in $required_set; do
		if [ ! -f "$DIR/$required" ]; then
			echo "restore: $DIR has no $required." >&2
			echo "  Refusing before anything is touched. This restore replaces the database" >&2
			echo "  and the file store together, so a backup missing either half would" >&2
			echo "  destroy the live copy of it and have nothing to write back." >&2
			exit 1
		fi
		grep -q "^artifact $required " "$DIR/manifest.txt" || {
			echo "restore: $required is present but the manifest does not list it." >&2
			echo "  Its checksum was therefore never verified. Refusing rather than" >&2
			echo "  replacing the live data with a file nothing vouches for." >&2
			exit 1
		}
	done
	echo "  $checked artifact(s) verified against the manifest"
}

# The collections the MANIFEST covers, one name per line, each validated.
#
# Read from the manifest rather than by globbing `qdrant/*.snapshot`, because
# only manifest-listed files have had their checksums verified. A stray
# snapshot - two backup directories merged by hand, an aborted sync - would
# otherwise be recovered over a live collection on the strength of nothing but
# its filename.
manifest_collections() {
	while read -r kind path _size _rest; do
		[ "$kind" = "artifact" ] || continue
		case "$path" in
		qdrant/*.snapshot) ;;
		*) continue ;;
		esac
		coll=${path#qdrant/}
		coll=${coll%.snapshot}
		if [ "$(safe_name "$coll")" != "ok" ]; then
			echo "restore: '$path' is not a name this can use as a collection." >&2
			echo "  Only letters, digits, dot, dash and underscore are accepted. Backups" >&2
			echo "  get copied and renamed by hand; a name outside that set would break" >&2
			echo "  out of the quoting used to place the file inside the container." >&2
			exit 1
		fi
		echo "$coll"
	done < "$DIR/manifest.txt"
}

head_report() {
	HEAD=$(manifest_value alembic_head)
	VERDICT=$(classify_head "$HEAD")
	GITREV=$(manifest_value git_rev)

	case "$VERDICT" in
	known)
		echo "  Schema: the dump ends at migration $HEAD, which this checkout has."
		echo "  Any migrations added since then run automatically on the next boot."
		;;
	unknown)
		echo ""
		echo "  !! THE DUMP IS AHEAD OF THIS CODE."
		echo ""
		echo "     It ends at migration $HEAD, and no migration in"
		echo "     apps/backend/alembic/versions/ declares that revision. The backend"
		echo "     migrates before it serves, and it cannot step forward from a"
		echo "     revision it does not have - so after this restore the stack will"
		echo "     FAIL TO START with:"
		echo ""
		echo "       Can't locate revision identified by '$HEAD'"
		echo ""
		echo "     Nothing is damaged by that; the data is fine and the message is"
		echo "     simply the boot refusing to guess. The fix is to deploy the code"
		echo "     the backup came from, or newer:"
		echo "       manifest git_rev: $GITREV"
		echo ""
		echo "     The restore continues, because recovered data belongs on disk"
		echo "     whether or not the deployed code has caught up yet."
		echo ""
		;;
	*)
		echo "  Schema: the dump's migration could not be compared against this"
		echo "  checkout (head '$HEAD'). Check by hand that the deployed code is at"
		echo "  or after the backup's commit before expecting the stack to boot."
		;;
	esac
}

require_app_stopped() {
	offenders=$(running_app_containers)
	[ -n "$offenders" ] || return 0

	echo "restore: the application services are still running." >&2
	echo "" >&2
	echo "  Running now:" >&2
	printf '    %s\n' $offenders >&2
	echo "" >&2
	echo "  They hold open database connections (the database cannot be dropped while" >&2
	echo "  one exists), they write attachment files underneath the archive this is" >&2
	echo "  about to unpack, and a chat turn in flight would write into a database" >&2
	echo "  that is being replaced." >&2
	echo "" >&2
	echo "  Stop them - and ONLY them; postgres and qdrant must stay up, because the" >&2
	echo "  restore talks to them:" >&2
	echo "" >&2
	echo "    docker compose --env-file .env $(compose_files) stop $APP_SERVICES" >&2
	echo "" >&2
	echo "  Then run this again. Nothing has been changed." >&2
	exit 1
}

preflight() {
	# No curl here on purpose: every HTTP call this script makes runs inside a
	# container, so the host needs Docker and tar and nothing else.
	for tool in docker tar; do
		command -v "$tool" >/dev/null 2>&1 || {
			echo "restore: '$tool' is not on PATH, and the restore needs it." >&2
			exit 1
		}
	done
	docker_ info >/dev/null 2>&1 || {
		echo "restore: cannot reach the Docker daemon - is Docker running?" >&2
		exit 1
	}
}

# Everything the LATER steps depend on, checked before the FIRST one runs.
#
# A restore replaces three things in sequence and cannot roll back, so any
# condition that would stop step 4 or 5 has to be discovered before step 3
# drops the database. Each check here corresponds to a way this script has
# been able to leave a stack holding a restored database beside the previous
# generation's vectors and files.
deep_preflight() {
	# Qdrant answers on HTTP, through the helper that every later call uses.
	# `need_container` proves a container is RUNNING, which is a different
	# claim: a network the service name does not resolve on, or a helper image
	# that cannot be pulled, passes that and fails at the recover call - with
	# postgres already replaced.
	QDRANT_NETWORK=$(qdrant_network)
	if ! docker_ network inspect "$QDRANT_NETWORK" >/dev/null 2>&1; then
		echo "restore: no Docker network named '$QDRANT_NETWORK'." >&2
		echo "  The vectors are recovered from a helper container joined to the stack's" >&2
		echo "  own network, so that network has to exist." >&2
		exit 1
	fi
	if ! qdrant_curl -fsS -o /dev/null "$QDRANT_URL/collections"; then
		echo "restore: Qdrant does not answer at $QDRANT_URL from network '$QDRANT_NETWORK'." >&2
		echo "  The vectors are recovered over that HTTP API, so this would fail AFTER" >&2
		echo "  the database had already been replaced. No published port is involved -" >&2
		echo "  the likely causes are qdrant still starting, or the $CURL_IMAGE image" >&2
		echo "  being unavailable to pull." >&2
		exit 1
	fi

	# The attachment volume exists. `docker run -v <name>:/dst` CREATES a
	# missing volume, so a wrong project name would unpack the archive into a
	# brand-new volume nothing mounts, report success, and leave the real one
	# holding the previous generation. Only checked when the archive is going
	# to be unpacked at all.
	if [ "$(manifest_captures_attachments)" = "yes" ] &&
		! docker_ volume inspect "$ATTACHMENT_VOLUME" >/dev/null 2>&1; then
		echo "restore: no volume named '$ATTACHMENT_VOLUME'." >&2
		echo "  Volumes are prefixed with the compose project name. This run resolved the" >&2
		echo "  project as '$PROJECT'; set COMPOSE_PROJECT_NAME if the stack uses another." >&2
		echo "  Continuing would create an empty volume nothing reads and report success." >&2
		exit 1
	fi

	# The helper image is usable. It is only needed by the LAST step, which is
	# exactly why it is checked first: a host that cannot pull it would fail
	# after postgres and the vectors had both been replaced.
	if ! docker_ run --rm "$HELPER_IMAGE" true >/dev/null 2>&1; then
		echo "restore: cannot run the '$HELPER_IMAGE' helper image." >&2
		echo "  It unpacks the attachment archive in the final step. Pull it first" >&2
		echo "  ('docker pull $HELPER_IMAGE') rather than discovering this after the" >&2
		echo "  database has been replaced." >&2
		exit 1
	fi
}

# --- plan --------------------------------------------------------------------

plan() {
	preflight
	echo "== restore plan =="
	echo "  source           $DIR"
	echo "  taken            $(manifest_value created_utc)"
	echo "  from commit      $(manifest_value git_rev)"
	echo "  compose project  $PROJECT"
	echo ""
	verify_backup
	head_report
	echo ""
	echo "  Would do, in order:"
	echo "    1. refuse unless $APP_SERVICES are stopped (stores stay up)"
	echo "    2. refuse unless qdrant answers on the stack network, the attachment"
	echo "       volume exists, both helper images run, and this server's pg_restore"
	echo "       can read the dump - all before any write"
	echo "    3. postgres: dropdb --force + createdb + pg_restore into '$PG_DB'"
	echo "    4. qdrant: recover each collection the manifest covers (overwrites it)"
	manifest_collections | sed 's/^/         /'
	if [ "$(manifest_captures_attachments)" = "yes" ]; then
		echo "    5. attachments: empty $ATTACHMENT_VOLUME, then unpack attachments.tar into it"
	else
		echo "    5. attachments: SKIPPED - this backup captured none"
		echo "       (storage_backend: $(manifest_value storage_backend); the volume is"
		echo "       left untouched, and the bucket needs its own restore)"
	fi
	echo ""
	deep_preflight
	echo "  Step 2 would pass: qdrant answers at $QDRANT_URL from network"
	echo "  '$QDRANT_NETWORK', $ATTACHMENT_VOLUME exists, and both helper images run."
	echo ""
	offenders=$(running_app_containers)
	if [ -n "$offenders" ]; then
		echo "  Right now step 1 would REFUSE - these are still up:"
		printf '    %s\n' $offenders
	else
		echo "  Step 1 would pass: no application container of this project is running."
	fi
	echo ""
	echo "  Nothing was changed. Run 'make restore DIR=$DIR' to do it."
}

# --- run ---------------------------------------------------------------------

run() {
	preflight
	echo "== 1/5 verify the backup =="
	verify_backup
	head_report

	echo "== 2/5 check the application is down =="
	require_app_stopped
	echo "  no application container of project '$PROJECT' is running"

	PG_CID=$(need_container postgres)
	QDRANT_CID=$(need_container qdrant)
	deep_preflight
	echo "  qdrant answers on HTTP, the attachment volume exists, the helper image runs"

	echo "== 3/5 postgres =="
	# THE LAST CHANCE TO STOP. This server reads the archive here, while the
	# live database is still intact - so an archive it cannot parse is caught
	# now rather than after the drop. It covers the version mismatch that no
	# other check can: a dump taken on a newer PostgreSQL is refused by an older
	# pg_restore, and without this that refusal arrives one command too late.
	docker_ exec -i "$PG_CID" pg_restore --list > /dev/null < "$DIR/postgres.dump" || {
		echo "restore: this server's pg_restore cannot read postgres.dump." >&2
		echo "  Nothing has been changed. The usual cause is a dump taken on a NEWER" >&2
		echo "  PostgreSQL than the container now running - archives are not readable" >&2
		echo "  backwards. Bring the postgres image up to the version the backup came" >&2
		echo "  from and run this again." >&2
		exit 1
	}
	echo "  postgres.dump is readable by this server - proceeding"

	# Drop and recreate rather than restoring over the top. `pg_restore --clean`
	# leaves behind anything the dump does not mention (a table added by newer
	# code, a row in a table the dump has no entry for), which is how a restore
	# quietly produces a database that matches neither generation. An empty
	# database restored into is exactly the dump and nothing else.
	#
	# --force terminates whatever sessions remain. The application is already
	# down by this point, so in practice this covers a stray `psql` rather than
	# the app - but a restore that fails on one forgotten shell is a restore
	# that fails at the worst moment.
	docker_ exec "$PG_CID" dropdb -U "$PG_USER" --if-exists --force "$PG_DB"
	docker_ exec "$PG_CID" createdb -U "$PG_USER" -O "$PG_USER" "$PG_DB"
	# --no-owner/--no-privileges so a dump taken under one role name restores
	# under another (deployments do not always share a database user).
	# --exit-on-error because pg_restore's default is to report errors, carry
	# on, and exit 0 - which reads as success over a partial restore.
	docker_ exec -i "$PG_CID" pg_restore -U "$PG_USER" -d "$PG_DB" \
		--no-owner --no-privileges --exit-on-error < "$DIR/postgres.dump"
	echo "  '$PG_DB' recreated and restored"

	echo "== 4/5 qdrant =="
	# Qdrant recovers a collection from a file IT can see, and the snapshots
	# directory is on the container's own writable layer rather than in the
	# qdrant_data volume - so the archive is copied in, recovered, and removed
	# again. Copying in beats the multipart upload endpoint because that one is
	# capped by `service.max_request_size_mb` (32 MB by default), which a real
	# collection passes long before anyone thinks to raise it.
	#
	# Recovering a collection that does not exist here is fine - it is created.
	# That is not an assumption: the API description shipped inside the image
	# (/qdrant/static/openapi.json, "Recover from a snapshot") states it, which
	# is what makes a restore onto a FRESH host work at all.
	found=0
	for coll in $(manifest_collections); do
		found=$((found + 1))
		snap="$DIR/qdrant/$coll.snapshot"
		sha=$(sha256_of "$snap")
		remote="/qdrant/snapshots/$coll/restore-in-progress.snapshot"

		docker_ exec "$QDRANT_CID" mkdir -p "/qdrant/snapshots/$coll"
		# Streamed in on stdin rather than `docker cp`. Both ends of a `docker
		# cp` argument are paths, and on Windows the host side needs the shell's
		# path translation while the container side needs it suppressed - there
		# is no setting that does both. A redirect is handled by the shell
		# before Docker sees anything, so it is correct on every host.
		docker_ exec -i "$QDRANT_CID" sh -c "cat > '$remote'" < "$snap"
		# The checksum is verified by Qdrant against the file it is about to
		# read, so a copy damaged in transit is refused rather than recovered.
		#
		# The status is captured so the copy can be removed either way. On
		# failure `set -e` would otherwise abort before the cleanup and leave a
		# full-size archive on the container's writable layer - outside every
		# volume, reclaimed by nothing short of recreating the container.
		rc=0
		printf '%s' "{\"location\":\"file://$remote\",\"priority\":\"snapshot\",\"checksum\":\"$sha\"}" \
			| qdrant_curl_stdin -fsS -o /dev/null -X PUT \
				-H 'content-type: application/json' -d @- \
				"$QDRANT_URL/collections/$coll/snapshots/recover?wait=true" || rc=$?
		docker_ exec "$QDRANT_CID" rm -f "$remote" || true
		if [ "$rc" -ne 0 ]; then
			echo "restore: Qdrant refused to recover '$coll' (curl exit $rc)." >&2
			echo "  The database has already been replaced; the vector store is part-way." >&2
			echo "  Fix the cause and re-run this restore - it is safe to repeat." >&2
			exit 1
		fi
		echo "  '$coll' recovered"
	done
	if [ "$found" -eq 0 ]; then
		echo "  the backup holds no snapshots - leaving the vector store untouched"
	fi

	# Collections the live store has and the backup does not are LEFT ALONE, and
	# said out loud. They are a real inconsistency - their vectors carry doc and
	# record ids pointing at rows the restored database no longer has, so they
	# answer retrieval with text from a generation that is gone. Deleting them
	# automatically is the worse failure though: a backup that legitimately
	# holds fewer collections (taken before an embedder change, say) would then
	# destroy live data the operator never asked this command to touch. So it
	# reports and leaves the decision where it belongs.
	extra=""
	for live in $(collection_names "$(qdrant_curl -fsS "$QDRANT_URL/collections" || true)"); do
		case " $(manifest_collections | tr '\n' ' ') " in
		*" $live "*) ;;
		*) extra="$extra $live" ;;
		esac
	done
	if [ -n "$extra" ]; then
		echo ""
		echo "  NOTE - collections present here but NOT in the backup:$extra"
		echo "  They were left untouched. Their vectors reference database rows the"
		echo "  restore has just replaced, so retrieval against them can surface text"
		echo "  from the previous generation. Drop any that should not survive - from"
		echo "  inside the network, since the port may not be published:"
		for name in $extra; do
			echo "    docker run --rm --network $QDRANT_NETWORK $CURL_IMAGE \\"
			echo "      -fsS -X DELETE $QDRANT_URL/collections/$name"
		done
	fi

	echo "== 5/5 attachments =="
	if [ "$(manifest_captures_attachments)" != "yes" ]; then
		# REFUSED, not skipped quietly. This step empties the volume before it
		# unpacks; with nothing to unpack it would delete whatever is there and
		# write nothing back. On a deployment that has since been switched to
		# local storage, that is the uploaded files.
		echo "  SKIPPED - this backup did not capture the attachment volume"
		echo "  (storage_backend: $(manifest_value storage_backend))"
		echo ""
		echo "  Nothing was emptied. The step that runs here REPLACES the volume, so"
		echo "  running it against a backup that holds no archive would delete the"
		echo "  files currently in it and put nothing back."
		echo ""
		echo "  The uploaded bytes for that backup are in object storage. Restore the"
		echo "  bucket with the provider's own tooling, to the state it was in at"
		echo "  $(manifest_value created_utc), or the attachment rows just restored will"
		echo "  point at objects that are not there."
	else
		# Emptied before unpacking, not merged into. The database was just
		# replaced wholesale, so a file left over from the other generation has
		# no row pointing at it: it would sit there forever, counted by nothing
		# and reclaimed by nothing.
		docker_ run --rm -i -v "$ATTACHMENT_VOLUME:/dst" "$HELPER_IMAGE" \
			sh -c 'set -e; cd /dst; find . -mindepth 1 -maxdepth 1 -exec rm -rf {} +; tar -xf - -C /dst' \
			< "$DIR/attachments.tar"
		echo "  $ATTACHMENT_VOLUME replaced from attachments.tar"

		# The reverse mismatch: the archive exists, but this deployment now
		# reads its files from a bucket. Unpacking is harmless (nothing reads
		# the volume) and reporting it is not - a restore that looked complete
		# while every attachment 404s is worth one line of warning.
		if [ "$(storage_captures_attachments "$STORAGE_BACKEND")" != "yes" ]; then
			echo ""
			echo "  NOTE - this deployment has STORAGE_BACKEND=$STORAGE_BACKEND, so nothing"
			echo "  reads that volume. The files were unpacked but will not be served;"
			echo "  the bucket is what needs restoring here."
		fi
	fi

	echo ""
	echo "Restore complete."
	echo ""
	echo "  Start the application services - the backend applies any pending"
	echo "  migrations before it serves, so nothing else is needed:"
	echo ""
	echo "    make up"
	echo ""
	echo "  The background job queue was not touched, so any jobs queued before the"
	echo "  restore still run. They reference rows by id, so one whose row the"
	echo "  restore replaced simply finds nothing and gives up - no cleanup needed."
	echo ""
	head_report
}

# --- dispatch ----------------------------------------------------------------

case "${1:-}" in
safe-name)
	safe_name "${2:-}"
	;;
collection-names)
	# The body arrives on STDIN, not as an argument - see the same dispatch in
	# scripts/backup.sh for why (a JSON document is mostly quote characters).
	collection_names "$(cat)"
	;;
network-name)
	compose_network_name
	;;
storage-captures-attachments)
	storage_captures_attachments "${2:-}"
	;;
compose-files)
	compose_files
	echo ""
	;;
attachments-plan)
	# What step 5 would do for this backup: "restore" or "skip". Reads only the
	# manifest, so the rule that decides whether a volume gets emptied can be
	# checked against a fixture rather than by emptying one.
	if [ -z "${2:-}" ]; then
		echo "usage: restore.sh attachments-plan <backup-directory>" >&2
		exit 2
	fi
	DIR=$2
	[ -f "$DIR/manifest.txt" ] || {
		echo "restore: $DIR has no manifest.txt." >&2
		exit 1
	}
	if [ "$(manifest_captures_attachments)" = "yes" ]; then
		echo "restore"
	else
		echo "skip"
	fi
	;;
classify-running)
	shift
	classify_running "$@"
	;;
classify-head)
	classify_head "${2:-}" "${3:-}"
	;;
plan)
	if [ -z "${2:-}" ]; then
		echo "usage: restore.sh plan <backup-directory>" >&2
		exit 2
	fi
	DIR=$2
	plan
	;;
verify)
	# Deliberately touches neither Docker nor the stack: an old backup on a
	# shelf is worth re-checking long before there is any intention of
	# restoring it, and the answer should not depend on a daemon being up.
	if [ -z "${2:-}" ]; then
		echo "usage: restore.sh verify <backup-directory>" >&2
		exit 2
	fi
	DIR=$2
	echo "== verify $DIR =="
	verify_backup
	head_report
	;;
collections)
	# Exactly the list the restore would recover, produced by exactly the code
	# that produces it - so the manifest-not-glob rule and the name guard can
	# be checked against a hostile directory without one being restored.
	if [ -z "${2:-}" ]; then
		echo "usage: restore.sh collections <backup-directory>" >&2
		exit 2
	fi
	DIR=$2
	[ -f "$DIR/manifest.txt" ] || {
		echo "restore: $DIR has no manifest.txt." >&2
		exit 1
	}
	manifest_collections
	;;
'')
	echo "usage: restore.sh <backup-directory> | plan <dir> | verify <dir> | classify-* ..." >&2
	echo "  Backups are the timestamped directories 'make backup' writes." >&2
	exit 2
	;;
*)
	DIR=$1
	run
	;;
esac
