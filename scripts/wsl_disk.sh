#!/bin/sh
# Report on, and reclaim, the Windows disk file that backs Docker's storage.
# Invoked by `make disk` (read-only) and `make compact` (reclaims). Both are
# no-ops with an explanatory message on any host that is not Windows.
#
# WHAT THIS IS ABOUT. On Windows, Docker keeps every image, every build-cache
# entry and every volume inside ONE file - a virtual disk, `docker_data.vhdx`.
# Docker sees it as a hard drive; Windows sees a single large file. That file
# grows when Docker stores more, and here is the part that surprises people:
#
#   IT NEVER SHRINKS ON ITS OWN.
#
# Deleting an image frees space INSIDE the virtual drive, so Docker reports the
# space as available, but the Windows file stays exactly as large as it ever
# was. The freed blocks become a hole inside the file that nothing reuses until
# Docker happens to need that much room again. Compacting is the separate step
# that collapses those holes and hands the space back to Windows.
#
# Docker Desktop used to do this automatically. That was disabled in 4.59.0
# ("Temporarily disabled VHDX compaction for the WSL data disk to improve
# stability") and is still off, so on a machine that builds images regularly the
# file only ratchets upward. One measured example: a file of 65.6 GB against
# 36.3 GB of live content, i.e. 25 GB of pure hole.
#
# THE TWO NUMBERS THAT MATTER, and they are easy to confuse:
#
#   "File on disk"  - what Windows sees. What is actually consumed.
#   "Used inside"   - what Docker sees. The live content.
#
# The gap between them is dead space, and it is the ONLY thing compaction can
# return. This is why order matters and why compacting on its own disappoints:
# compaction recovers what was already deleted, so a prune must come first or
# there is no hole to collapse. `make compact` therefore prunes, then compacts.
#
# WHY NOT FOLD THIS INTO `make prune`. Pruning is fast, needs no privileges and
# is safe while the stack is running. Compaction is the opposite on every count:
# it needs Docker Desktop fully stopped, it needs Administrator, it takes
# minutes, and it exists only on Windows. Joining them would mean every routine
# cleanup stopped the stack and raised a privilege prompt, usually to reclaim
# nothing - because normal use GROWS the file, and growth is not something
# compaction can help with. They run on different schedules by nature.
#
# ON `Optimize-VHD`. The commonly cited tool needs the Hyper-V role, which is
# absent from Windows Home. `diskpart` ships in every edition and does the same
# job through a different system library, which is what this script uses.
#
# THE OTHER THING REPORTED HERE IS WHO OWNS THE FILE, which sounds unrelated and
# is not. Moving the disk file to another drive is a supported setting, and the
# usual ways of doing it leave the new location owned by the Administrators
# group, or copy the file without its access rules. Either one produces a
# relocation that works perfectly until the machine is restarted and then
# refuses to start Docker at all - so the check belongs with the tool that knows
# where the file lives. The reasoning is at `perm_note` below.

set -e

MODE="${1:-report}"
# Messages name the make target the reader typed, not this script's internal mode.
if [ "$MODE" = "report" ]; then
	TARGET="disk"
else
	TARGET="compact"
fi

# Every Windows-side query goes through here. -NonInteractive so a stray prompt
# fails instead of hanging a make target.
#
# NOTE the exit status: this pipes into tr, so the status belongs to tr and is
# always 0. That is fine for reading a value, and WRONG for anything whose
# failure must be noticed - use ps_run below for those, or a silent failure
# reads as success and the caller reports work it never did.
ps_exec() {
	powershell.exe -NoProfile -NonInteractive -Command "$1" 2>/dev/null | tr -d '\r'
}

# Same, but the caller sees the real exit status. No pipe, and the Windows-side
# status is passed back explicitly because PowerShell's own exit code reflects
# the script block rather than the program it ran.
ps_run() {
	powershell.exe -NoProfile -NonInteractive -Command "$1
	  exit \$LASTEXITCODE" 2>/dev/null
}

# PowerShell single-quoted strings escape a quote by doubling it. Paths come
# from Docker's settings file rather than from a person, but a Windows user name
# with an apostrophe is ordinary, and unescaped it terminates the string early:
# the command then fails, ps_exec returns empty, and the caller reports a
# confident wrong answer instead of an error.
ps_quote() { printf '%s' "$1" | sed "s/'/''/g"; }

# --- host guard -------------------------------------------------------------
# Reaching powershell.exe is the practical test: it succeeds under Git Bash and
# under WSL on a Windows host, and fails everywhere this script is meaningless.
if ! command -v powershell.exe >/dev/null 2>&1; then
	echo "make $TARGET: skipped - this is a Windows-only maintenance command."
	echo ""
	echo "  Docker stores everything in a single virtual-disk FILE only on"
	echo "  Windows. On Linux and macOS hosts there is no such file to compact:"
	echo "  Linux stores layers directly on the filesystem, and macOS manages"
	echo "  its own disk image. 'make prune' is the whole story on those hosts."
	echo ""
	echo "  For how much room the machine has left and which of the stack's data"
	echo "  volumes is using it, run 'make host-disk' - that one works everywhere."
	exit 0
fi

# --- locate the disk file ---------------------------------------------------
# The location is a Docker Desktop setting, so read it rather than assuming.
# CustomWslDistroDir is absent on a default install, where the disk sits under
# the local app-data directory instead.
VHDX=$(ps_exec '
  $s = Join-Path $env:APPDATA "Docker\settings-store.json"
  $dir = $null
  if (Test-Path $s) { try { $dir = (Get-Content $s -Raw | ConvertFrom-Json).CustomWslDistroDir } catch {} }
  if (-not $dir) { $dir = Join-Path $env:LOCALAPPDATA "Docker\wsl" }
  Join-Path $dir "disk\docker_data.vhdx"
')

if [ -z "$VHDX" ]; then
	echo "make $TARGET: could not determine the Docker disk image location."
	exit 1
fi

VHDX_PS=$(ps_quote "$VHDX")

HOST_BYTES=$(ps_exec "(Get-Item '$VHDX_PS' -ErrorAction SilentlyContinue).Length")
if [ -z "$HOST_BYTES" ]; then
	echo "make $TARGET: no disk image found at:"
	echo "  $VHDX"
	echo ""
	echo "  Docker Desktop may never have started, or may use a different backend."
	exit 1
fi

# Live usage is read from inside the virtual machine, so it needs Docker up.
# Three traps here, each of which produced a WRONG number rather than an error:
#
#  1. Docker Desktop parks the data-disk mount when it goes idle, so a cold read
#     finds nothing. One trivial command wakes it first.
#  2. Under Git Bash, any argument that looks like a Unix path is rewritten into
#     a Windows path before the Windows-side program sees it. MSYS_NO_PATHCONV
#     turns that off; without it the query fails while still exiting 0.
#  3. Asking df for a path that is not mounted makes it answer about the closest
#     mount instead - the 132 MB root - which reads as plausible and is wrong.
#     So list every mount and match the mount point exactly.
# -P keeps each mount on ONE line: without it a long device name wraps and the
# used-bytes column shifts, quietly reporting the free figure as the used one.
read_used() {
	MSYS_NO_PATHCONV=1 wsl.exe -d docker-desktop -e df -B1 -P 2>/dev/null \
		| tr -d '\0' | awk '$NF == "/mnt/docker-desktop-disk" { print $3; exit }' || true
}

# The wake has to TOUCH STORAGE. A command that only lists containers is served
# without going near the data disk and leaves it parked (measured: still absent
# a second later), whereas asking for storage usage remounts it. Retried once,
# because the remount completes a moment after the command returns.
docker system df >/dev/null 2>&1 || true
USED_BYTES=$(read_used)
if [ -z "$USED_BYTES" ]; then
	sleep 2
	USED_BYTES=$(read_used)
fi

fmt_gb() { awk -v b="$1" 'BEGIN { printf "%.1f GB", b / 1073741824 }'; }

# Reports who owns the disk file, because the wrong answer breaks Docker in a
# way that takes a reboot to show up.
#
# Windows gives every virtual machine its own account, and grants that account
# access to each disk file the machine opens. The account changes identity when
# the machine restarts, so the grant has to be rewritten on the way back up -
# and rewriting an access rule requires OWNING the file. Ownership is therefore
# the entire test: an owner can always grant itself whatever else it needs.
#
# Administrator rights do not substitute. An ordinary terminal runs with that
# group membership disabled, so it confers nothing here even for an account that
# could elevate. A tree left owned by Administrators - which is what creating it
# from an elevated shell does - starts Docker once, successfully, and then fails
# on the next boot with:
#
#   Wsl/Service/CreateInstance/MountDisk/HCS/E_ACCESSDENIED
#
# A second dialog about a vpnkit handshake and a "bad magic string" follows; it
# is that same message being read where a handshake was expected, not a separate
# fault. Fix the access rules and both go away.
perm_note() {
	perm_info=$(ps_exec "\$a = Get-Acl '$VHDX_PS' -ErrorAction SilentlyContinue
	  if (\$a) {
	    \$me = [Security.Principal.WindowsIdentity]::GetCurrent().Name
	    \$root = Split-Path (Split-Path '$VHDX_PS' -Parent) -Parent
	    '{0}|{1}|{2}' -f \$a.Owner, \$me, \$root
	  }")
	perm_owner=$(printf '%s' "$perm_info" | cut -d'|' -f1)
	perm_me=$(printf '%s' "$perm_info" | cut -d'|' -f2)
	perm_root=$(printf '%s' "$perm_info" | cut -d'|' -f3)

	if [ -z "$perm_owner" ] || [ -z "$perm_me" ]; then
		echo "  Permissions   unknown   <- the owner could not be read"
		return
	fi
	# Windows account names are not case-sensitive.
	if [ "$(printf '%s' "$perm_owner" | tr 'A-Z' 'a-z')" = "$(printf '%s' "$perm_me" | tr 'A-Z' 'a-z')" ]; then
		echo "  Permissions   OK   <- owned by $perm_owner"
		return
	fi

	echo "  Permissions   WRONG   <- owned by $perm_owner, not $perm_me"
	echo ""
	echo "  Docker may run normally now and then FAIL TO START after the next"
	echo "  reboot, reporting:"
	echo "    Wsl/Service/CreateInstance/MountDisk/HCS/E_ACCESSDENIED"
	echo ""
	echo "  Windows gives each virtual machine its own account access to the disk"
	echo "  files it opens, and that account changes identity when the machine"
	echo "  restarts. Renewing the grant requires OWNING the file. Administrator"
	echo "  rights do not substitute - an ordinary terminal runs with that group"
	echo "  membership disabled."
	echo ""
	# The suggested target is the folder holding the disk image, two levels up
	# from the file. Refuse to print it if that resolves to a drive root: these
	# are recursive ownership changes, and aimed at C:\ one of them would walk
	# the whole system drive. A path that shallow means something is unusual
	# enough to deserve a person's eyes rather than a command to paste.
	if [ -z "$perm_root" ] || printf '%s' "$perm_root" | grep -qE '^[A-Za-z]:\\?$'; then
		echo "  Fix by taking ownership of the folder that holds the disk image,"
		echo "  then starting Docker Desktop. No command is printed here because"
		echo "  the folder resolves to a drive root, and changing ownership across"
		echo "  a whole drive would do far more damage than the fault. The file is:"
		echo "    $VHDX"
		return
	fi
	echo "  Fix from an ADMINISTRATOR terminal, then start Docker Desktop:"
	echo "    icacls \"$perm_root\" /setowner \"$perm_me\" /T /C"
	echo "    icacls \"$perm_root\" /grant \"$perm_me:(OI)(CI)F\" /T /C"
}

echo "== Docker disk image =="
echo "  Location      $VHDX"
echo "  File on disk  $(fmt_gb "$HOST_BYTES")   <- what Windows is actually giving up"

if [ -z "$USED_BYTES" ] || [ "$USED_BYTES" = "0" ]; then
	echo "  Used inside   unknown   <- needs Docker Desktop running to read"
	# Runs even here: reading the owner is a filesystem question, and this is
	# exactly the state a machine is in when the permission fault has bitten.
	perm_note
	echo ""
	echo "  Start Docker Desktop and re-run to see how much is reclaimable."
	# Plain `if`, not `[ ... ] && exit`: under `set -e` a false test at the end
	# of an AND-OR list ends the script, which would report success wrongly.
	if [ "$MODE" = "report" ]; then
		exit 0
	fi
	echo "  Cannot verify the gap before compacting. Start Docker Desktop first."
	exit 1
fi

RECOVERABLE=$((HOST_BYTES - USED_BYTES))
if [ "$RECOVERABLE" -lt 0 ]; then
	RECOVERABLE=0
fi
PCT=$(awk -v r="$RECOVERABLE" -v h="$HOST_BYTES" 'BEGIN { printf "%.0f", (h > 0 ? r * 100 / h : 0) }')

echo "  Used inside   $(fmt_gb "$USED_BYTES")   <- what Docker's content really needs"
echo "  Recoverable   $(fmt_gb "$RECOVERABLE") ($PCT%)   <- dead space compaction can return"
perm_note
echo ""

# Ten gigabytes, or a quarter of the file. Below that the several minutes and
# the full stack restart are not worth it, and a prune usually is.
WORTH_IT=0
if [ "$RECOVERABLE" -ge 10737418240 ] || [ "$PCT" -ge 25 ]; then
	WORTH_IT=1
fi

if [ "$MODE" = "report" ]; then
	if [ "$WORTH_IT" -eq 1 ]; then
		echo "  Worth compacting. Run 'make compact' from an ADMINISTRATOR terminal."
		echo "  It stops Docker for a few minutes, so pick a moment between tasks."
	else
		echo "  Nothing worth reclaiming yet. Compaction pays off once this passes"
		echo "  about 10 GB, which normally follows a large prune or image cleanup."
		echo "  Routine cleanup is 'make prune', which needs no privileges."
	fi
	exit 0
fi

# --- compact ----------------------------------------------------------------
ELEVATED=$(ps_exec '([Security.Principal.WindowsPrincipal][Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)')
if [ "$ELEVATED" != "True" ]; then
	echo "make compact: needs Administrator."
	echo ""
	echo "  Compaction rewrites a virtual disk, which Windows only permits with"
	echo "  elevated rights. Open a terminal with 'Run as administrator' and run"
	echo "  'make compact' there. Nothing has been changed."
	exit 1
fi

echo "== 1/5 Pruning (compaction can only reclaim what is already deleted) =="
docker image prune -f
docker builder prune -f
# Through the same guard the make targets use, so no volume prune in this repo
# is unguarded. `soft` because a refusal must not abort a compaction that is
# otherwise fine - it just leaves less to reclaim. In practice this never fires
# here: reaching this point means Docker Desktop, which bundles a current engine.
sh scripts/prune_guard.sh volumes soft
echo ""

echo "== 2/5 Trimming the filesystem inside the disk =="
# Marks the freed blocks as unused so the compactor can find them. Reporting
# 0 B is normal and is not a failure - the blocks may already be marked.
# The absolute /sbin path is required ('wsl -e' does not search it), and
# MSYS_NO_PATHCONV keeps Git Bash from rewriting both paths on the way through.
# Captured rather than piped straight to the terminal: a pipeline's status is
# the LAST command's, so a wrong distro name would look identical to a clean run.
TRIM=$(MSYS_NO_PATHCONV=1 wsl.exe -d docker-desktop -e /sbin/fstrim -v /mnt/docker-desktop-disk 2>/dev/null \
	| tr -d '\0') || true
if [ -n "$TRIM" ]; then
	echo "  $TRIM"
else
	echo "  (nothing reported; the blocks may already be marked - continuing)"
fi
echo ""

echo "== 3/5 Stopping Docker Desktop =="
docker desktop stop >/dev/null 2>&1 || true
sleep 8
wsl.exe --shutdown >/dev/null 2>&1 || true
sleep 5

# The file must be genuinely released. An "is it attached" query answers False
# while the virtual machine still holds it, so test the only thing that cannot
# lie: taking the file exclusively. If that succeeds, nothing else has it open.
# Empty output counts as "not released", so a failed probe stops rather than
# waving the compactor through.
check_released() {
	ps_exec "try { \$fs=[IO.File]::Open('$VHDX_PS','Open','Read','None'); \$fs.Close(); 'yes' } catch { 'no' }"
}
if [ "$(check_released)" != "yes" ]; then
	echo "  Docker is still holding the disk image open."
	echo "  The prune above did run and Docker is now stopped, but the disk image"
	echo "  itself was not touched. Quit Docker Desktop from the tray, then run"
	echo "  'make compact' again."
	exit 1
fi
echo "  Released."
echo ""

echo "== 4/5 Compacting (a few minutes; do not interrupt) =="
BEFORE="$HOST_BYTES"
SCRIPT_WIN=$(ps_exec 'Join-Path $env:TEMP "compact_docker_vhdx.txt"')
SCRIPT_PS=$(ps_quote "$SCRIPT_WIN")

# Written fresh every run and then verified. An unwritable path fails silently,
# and handing diskpart a stale file from an earlier run would point it at the
# wrong image - so the file is removed first and its contents checked after.
# -Encoding oem, not ascii: ascii replaces any non-ASCII character in the path
# with '?', after which diskpart selects nothing at all.
ps_exec "Remove-Item '$SCRIPT_PS' -ErrorAction SilentlyContinue"
ps_exec "Set-Content -Encoding oem -Path '$SCRIPT_PS' -Value @(
  'select vdisk file=\"$VHDX_PS\"',
  'attach vdisk readonly',
  'compact vdisk',
  'detach vdisk',
  'exit'
)"
if [ -z "$(ps_exec "if (Test-Path '$SCRIPT_PS') { (Get-Content '$SCRIPT_PS' -Raw) -match 'compact vdisk' }" | grep -i true)" ]; then
	echo "  Could not write the compaction script to a temporary file."
	echo "  Nothing was done to the disk image. Docker is stopped and can be restarted."
	exit 1
fi

# Re-probe immediately before running: Docker Desktop's background service can
# restart the virtual machine on its own between the check above and this point.
if [ "$(check_released)" != "yes" ]; then
	echo "  The disk image was re-opened while preparing. Nothing was done to it."
	echo "  Make sure Docker Desktop is not set to start automatically, then retry."
	exit 1
fi

# Attached read-only on purpose. It enables the full scan for reclaimable
# blocks, and it write-protects the disk meanwhile: if Windows offers to format
# the newly visible drive, accepting fails harmlessly instead of destroying it.
# Dismiss that prompt if it appears.
COMPACT_RC=0
ps_run "& diskpart /s '$SCRIPT_PS'" || COMPACT_RC=$?
ps_exec "Remove-Item '$SCRIPT_PS' -ErrorAction SilentlyContinue"

# diskpart stops at the first failing command, so a failed compaction never
# reaches its own 'detach vdisk' and would leave the image mounted in Windows -
# which blocks the next Docker start. Detach explicitly before reporting.
if [ "$COMPACT_RC" -ne 0 ]; then
	echo ""
	echo "  Compaction failed (exit $COMPACT_RC). Detaching the disk image so"
	echo "  Docker can start again."
	ps_exec "\$d = Join-Path \$env:TEMP 'detach_docker_vhdx.txt'
	  Set-Content -Encoding oem -Path \$d -Value @('select vdisk file=\"$VHDX_PS\"','detach vdisk','exit')
	  & diskpart /s \$d | Out-Null
	  Remove-Item \$d -ErrorAction SilentlyContinue"
	echo "  The disk image is unchanged. Start Docker Desktop when ready."
	exit 1
fi

AFTER=$(ps_exec "(Get-Item '$VHDX_PS' -ErrorAction SilentlyContinue).Length")
echo ""
echo "== 5/5 Result =="
if [ -z "$AFTER" ]; then
	# Refusing to guess: treating a missing size as zero would announce that the
	# entire image had been reclaimed.
	echo "  Compaction reported success, but the disk image could not be measured"
	echo "  afterwards. Check it by hand before starting Docker:"
	echo "    $VHDX"
	exit 1
fi
echo "  Before   $(fmt_gb "$BEFORE")"
echo "  After    $(fmt_gb "$AFTER")"
echo "  Freed    $(fmt_gb "$((BEFORE - AFTER))")"
echo ""
echo "  Start Docker Desktop when ready, then 'make up'."
