#!/bin/sh
# Verify that the running model server actually registered a GPU, and say what
# it found when it did not. Invoked by `make gpu-check`. Given a container it
# will RESTART that container once when it finds no GPU, so it is not a
# read-only command; the classify mode below is.
#
# Why this exists as a script rather than a make recipe: the interesting part
# is a parser, and a parser that cannot be run against a captured log is a
# parser nobody can check. Classify mode reads log text on standard input,
# prints its verdict, and exits 0 only for a GPU - so the matching rule can be
# exercised, and asserted on, without a container:
#
#   c=ollama-gpu
#   docker logs --since "$(docker inspect -f '{{.State.StartedAt}}' $c)" $c 2>&1 \
#     | sh scripts/gpu_check.sh classify
#
# Scope that read to one process, as above. Classify judges whatever text it is
# handed, so piping a WHOLE container log into it reproduces the very bug this
# script exists to fix (see rule 1 below).
#
# THE FAILURE THIS GUARDS AGAINST. The model server probes for GPUs at startup
# by spawning a helper process. On a busy host that probe hits its watchdog:
#
#   msg="llama-server GPU discovery watchdog timed out" ... error="context deadline exceeded"
#
# and the server then registers the CPU as its ONLY inference device for the
# rest of its life:
#
#   msg="inference compute" id=cpu library=cpu ... total="7.8 GiB"
#
# Every model afterwards loads at 100% CPU. Generation still works, so nothing
# fails outright - it is simply ten times slower, which reads as a performance
# regression rather than a broken deployment. A healthy start logs this instead:
#
#   msg="inference compute" id=0 library=CUDA compute=7.5 name=CUDA0 ...
#
# THREE RULES, EACH PAYING FOR ITSELF:
#
# 1. Only lines from the CURRENT process count. A container keeps its log
#    across restarts, so a whole-log search finds a healthy line from hours
#    ago and passes a server that is CPU-only right now. That is not a
#    hypothetical: it is the bug this script was written to fix, and it turned
#    a CPU-only deployment into a "confirmed healthy" one for an afternoon.
#    The window is the container's own start time, so a restart - including the
#    one this script performs - moves the window with it.
#
# 2. The evidence is the `inference compute` line and nothing else. The
#    discovery failures above name library paths like `cuda_v13`, so a search
#    for "CUDA" anywhere matches the very messages that report the failure.
#    Only a `library=` field on an `inference compute` line is read.
#
# 3. A discovery timeout is NOT failure evidence. Observed on a healthy start:
#    the server times out probing one library directory, then succeeds with
#    another and registers CUDA regardless. Timeouts are quoted as context when
#    the verdict is already CPU-only, never used to reach that verdict.
#
# Any GPU device in the window passes, rather than only the newest line: a host
# with several accelerators registers one line per device, and some register
# the CPU alongside them. The failure regime is the CPU as the ONLY device, so
# "no GPU device at all" is the thing to detect.
set -u

POLL_SECONDS=5
# How long to wait for the registration line before drawing a conclusion.
# Measured on a constrained host, from the server starting up: 38s and 76s on
# two healthy starts, 91s on one that landed on the CPU. The clock here starts
# when the script runs rather than when the server starts listening, which on
# that host costs another ten seconds or so of container start-up. The old 60s
# ceiling sat under the healthy maximum, so it could time out on a perfectly
# good start and restart the server for nothing.
GPU_CHECK_TIMEOUT=${GPU_CHECK_TIMEOUT:-120}
case $GPU_CHECK_TIMEOUT in
*[!0-9]* | '')
	# A `[ x -ge y ]` against a non-number errors rather than being false, and
	# the poll's exit condition would then never fire: an infinite wait.
	echo "GPU_CHECK_TIMEOUT must be a whole number of seconds (got '$GPU_CHECK_TIMEOUT')." >&2
	exit 2
	;;
esac

# `timeout` guards the corroboration calls below: a wedged model server would
# otherwise hang the failure path in silence, with no diagnosis printed. It is
# not universally present, so its absence degrades rather than fails.
if command -v timeout >/dev/null 2>&1; then
	HAS_TIMEOUT=1
else
	HAS_TIMEOUT=0
fi

# Accelerator backends the model server can name in `library=`. An unlisted
# value is reported rather than guessed at, so a backend nobody here has seen
# fails loudly with its own name in the message instead of passing quietly.
classify() {
	awk '
		/msg="inference compute"/ {
			seen = 1
			for (i = 1; i <= NF; i++) {
				if (substr($i, 1, 8) == "library=") {
					lib = substr($i, 9)
					if (tolower(lib) == "cpu") { cpu_seen = 1 }
					else if (tolower(lib) ~ /^(cuda|rocm|hip|vulkan|metal|oneapi|sycl|cann|musa)$/) {
						if (gpu == "") { gpu = lib }
					}
					else if (other == "") { other = lib }
				}
			}
		}
		END {
			# Exit status mirrors the verdict, so classify mode is assertable:
			# 0 only for a GPU, 1 for everything else.
			if (!seen)       { print "none";           exit 1 }
			if (gpu != "")   { print "gpu " gpu;       exit 0 }
			if (other != "") { print "unknown " other; exit 1 }
			if (cpu_seen)    { print "cpu";            exit 1 }
			print "none"; exit 1
		}
	'
}

if [ "${1:-}" = "classify" ]; then
	classify
	exit $?
fi

cid=${1:-}
if [ -z "$cid" ]; then
	echo "usage: gpu_check.sh <container-id> | gpu_check.sh classify < logfile" >&2
	exit 2
fi

# The container's own start time, with sub-second precision dropped. Only the
# fractional digits go: the trailing designator is preserved rather than
# rewritten to Z, since assuming UTC would shift the window HOURS earlier
# against an offset timestamp - straight back into a previous boot, which is
# the one remaining shape a false pass could take. Rounding to the second moves
# the edge earlier by under a second, and the previous process's last lines are
# whole seconds before this start, so nothing from either boot crosses over.
boot_log() {
	started=$(docker inspect -f '{{.State.StartedAt}}' "$cid" 2>/dev/null | sed 's/\.[0-9]*//')
	if [ -z "$started" ]; then
		return 1
	fi
	docker logs --since "$started" "$cid" 2>&1
}

# Poll until a GPU shows up or the clock runs out. It keeps waiting through a
# `cpu` reading rather than concluding at the first sight of one: on a host
# that registers the CPU before its accelerator, an early answer would restart
# a server that was seconds from succeeding.
poll_for_gpu() {
	elapsed=0
	while :; do
		# Captured in two steps on purpose: `verdict=$(boot_log | classify)`
		# would take the PIPELINE's status, which is awk's, so a container that
		# could not be inspected at all would read as "no device registered yet"
		# and be restarted on the strength of a diagnosis nobody made.
		if ! log=$(boot_log); then
			echo "unreachable"
			return 1
		fi
		verdict=$(printf '%s\n' "$log" | classify)
		case "$verdict" in
		gpu\ *)
			echo "$verdict"
			return 0
			;;
		esac
		[ "$elapsed" -ge "$GPU_CHECK_TIMEOUT" ] && break
		sleep "$POLL_SECONDS"
		elapsed=$((elapsed + POLL_SECONDS))
	done
	echo "$verdict"
	return 1
}

# What the resident models are actually running on. Corroboration only, never
# a verdict: this tier pins its embedding model to the CPU on purpose, so a
# model at 100% CPU is expected here and proves nothing by itself.
resident_models() {
	if [ "$HAS_TIMEOUT" = 1 ]; then
		timeout 10 docker exec "$cid" ollama ps 2>/dev/null
	else
		docker exec "$cid" ollama ps 2>/dev/null
	fi
}

# What the resident models are actually running on - a second opinion on the
# log's verdict, in both directions. It can only ever report, never decide,
# because "100% CPU" is the CORRECT answer for this tier's embedding model: it
# is pinned to the CPU deliberately, so a retrieval cannot evict the chat
# model. For a CHAT model the same reading would contradict a healthy verdict,
# which is worth saying out loud even though this cannot tell the two apart.
report_processors() {
	ps_out=$(resident_models)
	# A single line is the header by itself, which means nothing is loaded.
	if [ "$(echo "$ps_out" | grep -c .)" -lt 2 ]; then
		return 0
	fi
	case ${1:-failure} in
	failure)
		echo "  Resident models and what they are running on:"
		echo "$ps_out" | sed 's/^/    /'
		;;
	success)
		echo "$ps_out" | tail -n +2 | grep -q "100% CPU" || return 0
		echo "  Note: a resident model is running entirely on the CPU."
		echo "$ps_out" | sed 's/^/    /'
		echo "  Expected for the embedding model, which this tier pins to the CPU on"
		echo "  purpose. For a chat model it would contradict the verdict above - restart"
		echo "  the model server and re-check if one is listed."
		;;
	esac
}

explain_failure() {
	case "$1" in
	cpu)
		echo "  The model server registered the CPU as its ONLY inference device."
		echo "  GPU discovery timed out during startup, which happens when the host is"
		echo "  busy at boot. Every model will load at 100% CPU: roughly ten times"
		echo "  slower, with no error to show for it."
		timeout_line=$(boot_log | grep "GPU discovery watchdog timed out" | tail -n 1)
		if [ -n "$timeout_line" ]; then
			echo "  The last discovery timeout since this start:"
			echo "    $timeout_line"
		fi
		report_processors failure
		;;
	unreachable)
		echo "  Could not read the container's start time or its log. Is it still"
		echo "  running? 'make ps' shows the stack."
		;;
	none)
		echo "  No inference device was registered within ${GPU_CHECK_TIMEOUT}s of this start."
		echo "  Discovery may still be running, or the server may not have got as far as"
		echo "  listening. 'make logs' shows what it is doing."
		;;
	unknown\ *)
		echo "  The model server registered '${1#unknown }', which this check does not"
		echo "  recognise as an accelerator. If that backend is a GPU, add it to the"
		echo "  library list in scripts/gpu_check.sh; the check refuses to guess."
		;;
	esac
}

verdict=$(poll_for_gpu)
if [ $? -eq 0 ]; then
	echo "OK: the model server is on the GPU (${verdict#gpu })."
	report_processors success
	exit 0
fi

explain_failure "$verdict"

# A restart is worth trying only for the states a retry can actually change:
# discovery that lost a race with a busy host. It cannot teach the check a
# backend it does not know, and it cannot reach a container that could not be
# inspected - restarting on those would evict a warm model to no purpose and
# contradict the advice just printed.
case "$verdict" in
unknown\ * | unreachable)
	exit 1
	;;
esac

echo "GPU not registered yet - restarting the model server once..."
docker restart "$cid" >/dev/null || {
	echo "Could not restart the container."
	exit 1
}

# The restart moved the container's start time, so the window this re-check
# reads is the new process's own log and cannot inherit the old verdict.
verdict=$(poll_for_gpu)
if [ $? -eq 0 ]; then
	echo "OK: the model server is on the GPU after the restart (${verdict#gpu })."
	report_processors success
	exit 0
fi

echo "STILL NOT ON THE GPU after a restart."
explain_failure "$verdict"
echo "  What usually fixes it: wait for the machine to go idle, restart the model"
echo "  server once more, and run 'make gpu-check' again. If it never finds the"
echo "  GPU, work through the GPU checklist in README.md (driver, container"
echo "  toolkit, passthrough)."
echo "  Until then, treat any timing measurement from this stack as meaningless:"
echo "  it is measuring the CPU."
exit 1
