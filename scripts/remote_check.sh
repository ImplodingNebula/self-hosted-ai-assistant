#!/bin/sh
# Decide whether OLLAMA_URL names a model server this machine starts, and refuse
# the combinations that cannot work. Invoked by `make env-check` (which every
# up/rebuild/pull/warm target runs first) and by the model targets, which need
# to know whether there is a container to exec into at all.
#
# Why this exists as a script rather than a make recipe: the interesting part is
# a classifier, and a classifier that cannot be run against inputs is one nobody
# can check. Classify mode takes a URL as an argument, prints its verdict and
# exits 0, so the rule can be exercised without a stack, a .env, or a network:
#
#   sh scripts/remote_check.sh classify https://a-11434.proxy.example.net
#   sh scripts/remote_check.sh classify http://ollama:11434
#
# Usage:
#   sh scripts/remote_check.sh classify <url>     print "local" or "remote"
#   sh scripts/remote_check.sh classify-env       ... for OLLAMA_URL in ./.env
#   sh scripts/remote_check.sh check <profile>    warn or refuse for this setup
#
# WHAT "REMOTE" CHANGES. A local model server is a container this compose stack
# starts, sitting on a private network: the model targets reach it with `docker
# exec`, and it needs no credentials because nothing outside the network can
# address it. A remote one is neither. It is reached over HTTP like any other
# service, and because Ollama has no authentication of its own, the only thing
# between a published endpoint and the open internet is whatever reverse proxy
# was put in front of it. Both differences are silent failures if nobody checks:
# an exec into a container that was never started reports the wrong instance's
# state, and an unprotected endpoint works perfectly for its owner and for
# everyone else who finds it.
#
# THE HOST LIST BELOW EXISTS TWICE. The backend carries its own copy (the
# helpers in apps/backend/app/core/config.py) because it makes the same
# distinction on the request path. They are small, they are read by different languages, and
# sharing them would mean the shell parsing Python or the reverse - so they are
# duplicated deliberately, and each names the other. Change one, change both.
set -u

# Hosts that mean "the model server this stack starts". Anything else is remote.
# The three service names are the compose aliases; localhost and the loopback
# literals cover a server run beside the stack rather than inside it; and
# host.docker.internal is how a container reaches a server on its own host.
# KEEP IN SYNC with the backend's copy (see the note above).
LOCAL_HOSTS="ollama ollama-gpu ollama-cpu localhost 127.0.0.1 ::1 host.docker.internal"

# Where the client-side settings live. Overridable so the rules can be exercised
# against a fixture rather than the machine's real configuration.
ENV_FILE=${REMOTE_CHECK_ENV:-.env}

# The hostname alone: scheme, userinfo, port and path all removed, lowercased.
# Comparing anything more would make http://ollama:11434 and http://ollama/
# different answers to the same question. A value with no "://" has no host at
# all - "10.0.0.5:11434" or a stray sentence parses as a path, not an address -
# matching the backend's urlparse semantics so the two classifiers cannot give
# the same value opposite verdicts.
url_host() {
	case "${1:-}" in
	*://*) ;;
	*)
		printf '%s' ""
		return 0
		;;
	esac
	h=$(printf '%s' "${1:-}" | sed -e 's|^[A-Za-z][A-Za-z0-9+.-]*://||' -e 's|[/?#].*$||' -e 's|^.*@||')
	case "$h" in
	\[*)
		# A bracketed IPv6 literal. The brackets exist precisely because the
		# address itself contains colons, so a port cannot be cut at the first
		# one - the closing bracket is the boundary.
		h=${h#\[}
		h=${h%%\]*}
		;;
	*)
		h=${h%%:*}
		;;
	esac
	printf '%s' "$h" | tr 'ABCDEFGHIJKLMNOPQRSTUVWXYZ' 'abcdefghijklmnopqrstuvwxyz'
}

# The scheme, lowercased; empty when the URL carries none.
url_scheme() {
	printf '%s' "${1:-}" | sed -n 's|^\([A-Za-z][A-Za-z0-9+.-]*\)://.*|\1|p' \
		| tr 'ABCDEFGHIJKLMNOPQRSTUVWXYZ' 'abcdefghijklmnopqrstuvwxyz'
}

# An unset or empty URL classifies as local: that is the state of a fresh
# checkout, where the compose default applies and a local container is exactly
# what starts.
classify() {
	host=$(url_host "${1:-}")
	if [ -z "$host" ]; then
		echo "local"
		return 0
	fi
	for known in $LOCAL_HOSTS; do
		if [ "$host" = "$known" ]; then
			echo "local"
			return 0
		fi
	done
	echo "remote"
}

# One key's value out of the env file. The LAST uncommented assignment wins,
# matching how every other reader of this file treats a repeated key. A leading
# `#` is not whitespace, so commented lines never match the anchor and cannot
# be mistaken for settings. Carriage returns go first because this file is
# edited on Windows as often as not, and a trailing one would break every
# comparison below while being invisible on screen.
env_value() {
	[ -f "$ENV_FILE" ] || return 0
	value=$(tr -d '\r' < "$ENV_FILE" | sed -n "s/^[[:space:]]*$1=//p" | tail -n 1)
	# Quotes around a value belong to the file format, not to the setting.
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

check() {
	profile=${1:-}

	# No env file yet is a normal state, not a fault: `make env-check` can run
	# before one has been created. There is nothing to judge, so say nothing.
	[ -f "$ENV_FILE" ] || return 0

	url=$(env_value OLLAMA_URL)
	kind=$(classify "$url")
	scheme=$(url_scheme "$url")
	# Only ever tested for emptiness. The value is a credential and is never
	# printed, quoted back, or included in any message below.
	headers=$(env_value OLLAMA_AUTH_HEADERS)

	if [ "$kind" = "remote" ] && { [ "$profile" = "gpu" ] || [ "$profile" = "cpu" ]; }; then
		echo "REFUSING: OLLAMA_URL points at a remote model server, but PROFILE=$profile."
		echo "  That profile starts a local Ollama container nothing would use, and the"
		echo "  model targets would then act on that idle container instead of the server"
		echo "  actually serving this stack - reporting its models, and unloading nothing."
		echo "  Run the make targets with PROFILE=remote, or add a PROFILE=remote line to"
		echo "  .env to make it standing. See infra/remote-ollama/README.md."
		return 1
	fi

	if [ "$kind" = "local" ] && [ "$profile" = "remote" ]; then
		echo "REFUSING: PROFILE=remote, but OLLAMA_URL points at a local model server."
		echo "  PROFILE=remote starts no Ollama container, so the backend would spend the"
		echo "  whole session dialling a service that is not running."
		echo "  Point OLLAMA_URL at the remote address, or drop PROFILE=remote."
		return 1
	fi

	# Everything below WARNS and lets the command through. Both conditions are
	# things an operator may have decided on purpose (a plain-http endpoint on a
	# trusted link, a proxy that authenticates by network rather than by header),
	# and a setup that is merely questionable is not one to block.
	if [ "$kind" = "remote" ] && [ "$scheme" = "http" ]; then
		echo "WARNING: OLLAMA_URL is a remote address on plain http."
		echo "  A cloud pod's HTTP proxy normally answers http with a redirect to https,"
		echo "  and the backend does not follow redirects - it reports the redirect and"
		echo "  stops, rather than replaying the request (and its credentials) at an"
		echo "  address nobody chose. Use the https address unless the endpoint is known"
		echo "  to serve plain http."
	fi

	if [ "$kind" = "remote" ] && [ -z "$headers" ]; then
		echo "WARNING: OLLAMA_URL is remote and OLLAMA_AUTH_HEADERS is not set."
		echo "  Ollama has no authentication of its own, so unless a reverse proxy in"
		echo "  front of it checks a credential, that endpoint is open: anyone who finds"
		echo "  the URL can run generations on the GPU and pull or delete models."
		echo "  infra/remote-ollama/README.md has the proxy recipe and the header to set."
	fi

	return 0
}

case "${1:-}" in
classify)
	classify "${2:-}"
	;;
classify-env)
	classify "$(env_value OLLAMA_URL)"
	;;
check)
	check "${2:-}"
	exit $?
	;;
*)
	echo "usage: remote_check.sh classify <url> | classify-env | check <profile>" >&2
	exit 2
	;;
esac
