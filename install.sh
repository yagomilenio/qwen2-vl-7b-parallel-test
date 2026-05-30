#!/bin/sh
# shellcheck shell=dash
# shellcheck disable=SC2039  # local is non-POSIX
#
# Licensed under the MIT license
# <LICENSE-MIT or https://opensource.org/licenses/MIT>, at your
# option. This file may not be copied, modified, or distributed
# except according to those terms.

# This runs on Unix shells like bash/dash/ksh/zsh. It uses the common `local`
# extension. Note: Most shells limit `local` to 1 var per line, contra bash.

# Some versions of ksh have no `local` keyword. Alias it to `typeset`, but
# beware this makes variables global with f()-style function syntax in ksh93.
# mksh has this alias by default.
has_local() {
    # shellcheck disable=SC2034  # deliberately unused
    local _has_local
}

has_local 2>/dev/null || alias local=typeset

set -u

# lms config
APP_NAME="llmster"
APP_VERSION="0.0.15-2"
APP_VARIANT="full"
ZIP_EXT=".tar.gz"
CHECKSUM_EXT=".sha512"
ARTIFACT_DOWNLOAD_URL="llmster.lmstudio.ai/download"

if [ -n "${LMS_PRINT_VERBOSE:-}" ]; then
    PRINT_VERBOSE="$LMS_PRINT_VERBOSE"
else
    PRINT_VERBOSE=${INSTALLER_PRINT_VERBOSE:-0}
fi
if [ -n "${LMS_PRINT_QUIET:-}" ]; then
    PRINT_QUIET="$LMS_PRINT_QUIET"
else
    PRINT_QUIET=${INSTALLER_PRINT_QUIET:-0}
fi
if [ -n "${LMS_NO_MODIFY_PATH:-}" ]; then
    NO_MODIFY_PATH="$LMS_NO_MODIFY_PATH"
else
    NO_MODIFY_PATH=${INSTALLER_NO_MODIFY_PATH:-0}
fi

# Some Linux distributions don't set HOME
# https://github.com/astral-sh/uv/issues/6965#issuecomment-2915796022
get_home() {
    if [ -n "${HOME:-}" ]; then
        echo "$HOME"
        return
    fi

    if [ -n "${USER:-}" ]; then
        if command -v getent > /dev/null 2>&1; then
            getent passwd "$USER" | cut -d: -f6
            return
        fi
        if command -v dscl > /dev/null 2>&1; then
            dscl . -read "/Users/$USER" NFSHomeDirectory 2>/dev/null | awk '{print $2}'
            return
        fi
    fi

    if command -v getent > /dev/null 2>&1; then
        getent passwd "$(id -un)" | cut -d: -f6
        return
    fi
    if command -v dscl > /dev/null 2>&1; then
        dscl . -read "/Users/$(id -un)" NFSHomeDirectory 2>/dev/null | awk '{print $2}'
        return
    fi

    echo ""
}
INFERRED_HOME=$(get_home)

usage() {
    # print help (this cat/EOF stuff is a "heredoc" string)
    cat <<EOF
llmster-installer-$APP_VARIANT.sh

The installer for LM Studio daemon (llmster) v${APP_VERSION}

This script detects what platform you're on and fetches an appropriate archive from lmstudio.ai
then unpacks the binaries and installs them to

    \${HOME}/.lmstudio/bin (or \${HOME}/.cache/lm-studio/bin)

It will then add that dir to PATH by adding the appropriate line to your shell profiles.

USAGE:
    llmster-installer-$APP_VARIANT.sh [OPTIONS]

OPTIONS:
    -v, --verbose
            Enable verbose output

    -q, --quiet
            Disable progress output

        --no-modify-path
            Don't configure the PATH environment variable

    -h, --help
            Print help information
EOF
}

download_binary_and_run_installer() {
    downloader --check
    need_cmd uname
    need_cmd mktemp
    need_cmd chmod
    need_cmd mkdir
    need_cmd rm
    need_cmd tar
    need_cmd grep
    need_cmd cat

    for arg in "$@"; do
        case "$arg" in
            --help)
                usage
                exit 0
                ;;
            --quiet)
                PRINT_QUIET=1
                ;;
            --verbose)
                PRINT_VERBOSE=1
                ;;
            --no-modify-path)
                NO_MODIFY_PATH=1
                ;;
            *)
                OPTIND=1
                if [ "${arg%%--*}" = "" ]; then
                    err "unknown option $arg"
                fi
                while getopts :hvq sub_arg "$arg"; do
                    case "$sub_arg" in
                        h)
                            usage
                            exit 0
                            ;;
                        v)
                            # user wants to skip the prompt --
                            # we don't need /dev/tty
                            PRINT_VERBOSE=1
                            ;;
                        q)
                            # user wants to skip the prompt --
                            # we don't need /dev/tty
                            PRINT_QUIET=1
                            ;;
                        *)
                            err "unknown option -$OPTARG"
                            ;;
                        esac
                done
                ;;
        esac
    done

    if command -v check_llmster_is_not_running >/dev/null 2>&1; then
        check_llmster_is_not_running
    fi

    # check that `libatomic` is installed (required on Linux for Node.js 25+)
    if [ "$(uname -s)" = "Linux" ]; then
        if ! check_cmd ldconfig; then
            err "\"ldconfig\" must be available on your PATH before installing.

Please ensure ldconfig is installed and in your PATH, then run this again."
        fi
        if ! ldconfig -p 2>/dev/null | awk '$1=="libatomic.so.1"{found=1} END{exit !found}'; then
            err "Missing required dependency: \"libatomic\" is not installed on your system.

To install it:

    Debian/Ubuntu: sudo apt-get update && sudo apt-get install -y libatomic1
    Fedora/RHEL:  sudo dnf install -y libatomic

Afterwards, run this installer again."
        fi
    fi

    local _release_name
    _release_name="$(get_release_name)" || return 1
    local _artifact_name
    _artifact_name="$_release_name$ZIP_EXT"
    _zip_ext=${ZIP_EXT}

    # download the archive
    local _url="$ARTIFACT_DOWNLOAD_URL/$_artifact_name"
    local _dir
    _dir="$(ensure mktemp -d)" || return 1
    chmod 777 "$_dir"
    local _file="$_dir/input$_zip_ext"

    console_log "Downloading $APP_NAME $APP_VERSION $(uname -sm)" 1>&2
    console_log_verbose "  from $_url" 1>&2
    console_log_verbose "  to $_file" 1>&2

    ensure mkdir -p "$_dir"

    if ! downloader "$_url" "$_file"; then
      console_log "Failed to download llmster from '$_url'."
      console_log "Please check your network connection."
      console_log ""
      console_log "If this problem persists, please feel free to open an issue"
      console_log "in https://github.com/lmstudio-ai/lmstudio-bug-tracker"
      exit 1
    fi

    local _checksum_artifact_name="$_release_name$CHECKSUM_EXT"
    local _checksum_url="$ARTIFACT_DOWNLOAD_URL/$_checksum_artifact_name"
    local _checksum_file="$_dir/checksum"
    local _checksum_value

    if downloader "$_checksum_url" "$_checksum_file" "1"; then
        _checksum_value=$(cat "$_checksum_file")
    fi

    verify_checksum "$_file" "$_checksum_value"

    # Install logging is handled in the installer; legacy path prints inside install()
    ensure tar xf "$_file" -C "$_dir"

    install "$_dir"
    local _retval=$?
    if [ "$_retval" != 0 ]; then
        return "$_retval"
    fi

    ignore rm -rf "$_dir"

    return "$_retval"
}

get_release_name() {
    case "$(uname -sm)" in
        "Linux x86_64")
            local _bundle_suffix=""
            if [ "$APP_VARIANT" = "full" ]; then
                _bundle_suffix="$(get_linux_x86_full_bundle_suffix)"
                echo "${APP_VERSION}-linux-x64.full${_bundle_suffix}"
            else
                echo "${APP_VERSION}-linux-x64.${APP_VARIANT}"
            fi
            return 0
            ;;
        "Linux aarch64"|"Linux arm64")
            echo "${APP_VERSION}-linux-arm64.${APP_VARIANT}"
            return 0
            ;;
        "Darwin x86_64")
            echo "${APP_VERSION}-darwin-x64.${APP_VARIANT}"
            return 0
            ;;
        "Darwin arm64")
            echo "${APP_VERSION}-darwin-arm64.${APP_VARIANT}"
            return 0
            ;;
    esac
    err "no compatible downloads were found for your platform $(uname -sm)"
}

console_log() {
    if [ "0" = "$PRINT_QUIET" ]; then
        echo "$1"
    fi
}

console_log_verbose() {
    if [ "1" = "$PRINT_VERBOSE" ]; then
        echo "$1"
    fi
}

console_log_gray() {
    local gray
    local reset
    # Prefer "bright black" (often gray) if supported; fall back to white.
    gray=$(tput setaf 8 2>/dev/null || tput setaf 7 2>/dev/null || echo '')
    reset=$(tput sgr0 2>/dev/null || echo '')
    console_log "${gray}$1${reset}"
}

warn() {
    if [ "0" = "$PRINT_QUIET" ]; then
        local red
        local reset
        red=$(tput setaf 1 2>/dev/null || echo '')
        reset=$(tput sgr0 2>/dev/null || echo '')
        console_log "${red}WARN${reset}: $1" >&2
    fi
}

err() {
    if [ "0" = "$PRINT_QUIET" ]; then
        local red
        local reset
        red=$(tput setaf 1 2>/dev/null || echo '')
        reset=$(tput sgr0 2>/dev/null || echo '')
        console_log "${red}ERROR${reset}: $1" >&2
    fi
    exit 1
}

need_cmd() {
    if ! check_cmd "$1"
    then err "need '$1' (command not found)"
    fi
}

check_cmd() {
    command -v "$1" > /dev/null 2>&1
    return $?
}

assert_nz() {
    if [ -z "$1" ]; then err "assert_nz $2"; fi
}

# Run a command that should never fail. If the command fails execution
# will immediately terminate with an error showing the failing
# command.
ensure() {
    if ! "$@"; then err "command failed: $*"; fi
}

# This is just for indicating that commands' results are being
# intentionally ignored. Usually, because it's being executed
# as part of error handling.
ignore() {
    "$@"
}

# This wraps curl or wget. Try curl first, if not installed,
# use wget instead.
downloader() {
    # Check if we have a broken snap curl
    # https://github.com/boukendesho/curl-snap/issues/1
    _snap_curl=0
    if command -v curl > /dev/null 2>&1; then
      _curl_path=$(command -v curl)
      if echo "$_curl_path" | grep "/snap/" > /dev/null 2>&1; then
        _snap_curl=1
      fi
    fi

    # Check if we have a working (non-snap) curl
    if check_cmd curl && [ "$_snap_curl" = "0" ]
    then _dld=curl
    # Try wget for both no curl and the broken snap curl
    elif check_cmd wget
    then _dld=wget
    # If we can't fall back from broken snap curl to wget, report the broken snap curl
    elif [ "$_snap_curl" = "1" ]
    then
      console_log "curl installed with snap cannot be used to install $APP_NAME"
      console_log "due to missing permissions. Please uninstall it and"
      console_log "reinstall curl with a different package manager (e.g., apt)."
      console_log "See https://github.com/boukendesho/curl-snap/issues/1"
      exit 1
    else _dld='curl or wget' # to be used in error message of need_cmd
    fi

    # Determine effective quiet mode - use 3rd argument if provided, otherwise use global setting
    local _effective_quiet="$PRINT_QUIET"
    if [ -n "${3:-}" ]; then
        _effective_quiet="$3"
    fi

    local _curl_verbosity
    if [ "$_effective_quiet" = "1" ]; then
        _curl_verbosity="-s"
    else
        _curl_verbosity="--progress-bar"
    fi

    local _curl_columns_env=""
    if [ "$_effective_quiet" = "0" ]; then
        _curl_columns_env="COLUMNS=60"
    fi

    if [ "$1" = --check ]
    then need_cmd "$_dld"
    elif [ "$_dld" = curl ]; then
        if [ "$_curl_columns_env" != "" ]; then
            env "$_curl_columns_env" curl -SfL $_curl_verbosity "$1" -o "$2"
        else
            curl -SfL $_curl_verbosity "$1" -o "$2"
        fi
    elif [ "$_dld" = wget ]; then
        wget "$1" -O "$2"
    else err "Unknown downloader"   # should not reach here
    fi
}

verify_checksum() {
    local _file="$1"
    local _checksum_value="$2"
    local _calculated_checksum
    local _checksum_cmd=""

    if [ -z "$_checksum_value" ]; then
        return 0
    fi

    if check_cmd sha512sum; then
        _checksum_cmd="sha512sum"
    elif check_cmd gsha512sum; then
        _checksum_cmd="gsha512sum"
    elif check_cmd shasum; then
        _checksum_cmd="shasum"
    fi
    if [ -z "$_checksum_cmd" ] || ! check_cmd awk; then
        console_log "Skipping checksum verification (missing required commands)"
        return 0
    fi

    console_log "Verifying checksum..."
    if [ "$_checksum_cmd" = "shasum" ]; then
        _calculated_checksum="$("$_checksum_cmd" -a 512 "$_file" | awk '{printf $1}')"
    else
        _calculated_checksum="$("$_checksum_cmd" -b "$_file" | awk '{printf $1}')"
    fi
    if [ "$_calculated_checksum" != "$_checksum_value" ]; then
        err "checksum mismatch
            want: $_checksum_value
            got:  $_calculated_checksum"
    fi
}

detect_nvidia_driver_version_first_row() {
    # Skip detection without `timeout`. nvidia-smi can hang indefinitely in some
    # driver states, so we only probe when we can guard it.
    if ! check_cmd timeout; then
        echo ""
        return 0
    fi
    if ! check_cmd nvidia-smi; then
        echo ""
        return 0
    fi

    local _smi_output=""
    if _smi_output="$(timeout 5 nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null)"; then
        : "nvidia-smi succeeded"
    else
        _smi_output=""
    fi

    if [ -z "$_smi_output" ]; then
        echo ""
        return 0
    fi

    # Take the first line (multi-GPU systems return one line per GPU)
    local _first_line=""
    IFS= read -r _first_line <<EOF
$_smi_output
EOF

    # Trim leading/trailing spaces
    while [ "${_first_line# }" != "$_first_line" ]; do _first_line="${_first_line# }"; done
    while [ "${_first_line% }" != "$_first_line" ]; do _first_line="${_first_line% }"; done

    if [ -z "$_first_line" ]; then
        echo ""
        return 0
    fi

    echo "$_first_line"
}

normalize_driver_version() {
    local _version="$1"
    # Trim leading/trailing spaces
    while [ "${_version# }" != "$_version" ]; do _version="${_version# }"; done
    while [ "${_version% }" != "$_version" ]; do _version="${_version% }"; done

    if [ -z "$_version" ]; then
        echo ""
        return 0
    fi

    local _major=""
    _major="${_version%%.*}"
    local _rest=""
    _rest="${_version#*.}"
    if [ "$_rest" = "$_version" ]; then
        echo ""
        return 0
    fi

    local _minor=""
    _minor="${_rest%%.*}"
    local _patch=""
    if [ "$_minor" = "$_rest" ]; then
        _patch="0"
    else
        _patch="${_rest#*.}"
    fi

    if [ "${_patch#*.}" != "$_patch" ]; then
        echo ""
        return 0
    fi

    case "$_major" in ""|*[!0-9]*) echo ""; return 0 ;; esac
    case "$_minor" in ""|*[!0-9]*) echo ""; return 0 ;; esac
    case "$_patch" in ""|*[!0-9]*) echo ""; return 0 ;; esac

    echo "${_major}.${_minor}.${_patch}"
}

driver_version_gte() {
    local _version="$1"
    local _threshold="$2"

    if [ -z "$_version" ]; then
        return 1
    fi
    if [ -z "$_threshold" ]; then
        return 1
    fi

    local _version_major=""
    local _version_minor=""
    local _version_patch=""
    local _version_rest=""
    _version_major="${_version%%.*}"
    _version_rest="${_version#*.}"
    _version_minor="${_version_rest%%.*}"
    _version_patch="${_version_rest#*.}"

    local _threshold_major=""
    local _threshold_minor=""
    local _threshold_patch=""
    local _threshold_rest=""
    _threshold_major="${_threshold%%.*}"
    _threshold_rest="${_threshold#*.}"
    _threshold_minor="${_threshold_rest%%.*}"
    _threshold_patch="${_threshold_rest#*.}"

    if [ "$_version_major" -gt "$_threshold_major" ]; then
        return 0
    fi
    if [ "$_version_major" -lt "$_threshold_major" ]; then
        return 1
    fi

    if [ "$_version_minor" -gt "$_threshold_minor" ]; then
        return 0
    fi
    if [ "$_version_minor" -lt "$_threshold_minor" ]; then
        return 1
    fi

    if [ "$_version_patch" -ge "$_threshold_patch" ]; then
        return 0
    fi

    return 1
}

supports_cuda_12_4_or_newer() {
    local _driver_version_raw=""
    _driver_version_raw="$(detect_nvidia_driver_version_first_row)"
    if [ -z "$_driver_version_raw" ]; then
        return 1
    fi

    local _driver_version=""
    _driver_version="$(normalize_driver_version "$_driver_version_raw")"
    if [ -z "$_driver_version" ]; then
        return 1
    fi

    local _minimum_driver_version="550.54.14"
    if driver_version_gte "$_driver_version" "$_minimum_driver_version"; then
        return 0
    fi

    return 1
}

get_linux_x86_full_bundle_suffix() {
    if supports_cuda_12_4_or_newer; then
        echo "+cuda12"
    else
        echo ""
    fi
    return 0
}

install() {
    local _src_dir="$1"
    local _bootstrap_binary="$_src_dir/llmster"

    if [ ! -x "$_bootstrap_binary" ]; then
        err "Bootstrap binary not found at $_bootstrap_binary"
    fi

    console_log "Installing llmster..."

    # check that `libgomp` is installed
    if check_cmd ldconfig && check_cmd awk && [ "$(uname -s)" = "Linux" ]; then
        ldconfig -p 2>/dev/null | awk '$1=="libgomp.so.1"{f=1} END{exit !f}'
        local _awk_result=$?
        if [ "$_awk_result" -ne 0 ]; then
            console_log_gray "Warning: Could not detect libgomp. You might need to install libgomp"
        fi
    fi

    LMS_BOOTSTRAP_INSTALL_SH=1 ensure "$_bootstrap_binary" bootstrap

    # Best-effort PATH helper using the installed lms
    local _lmstudio_home_ptr="$INFERRED_HOME/.lmstudio-home-pointer"
    local _lmstudio_home="$INFERRED_HOME/.lmstudio"
    if [ -f "$_lmstudio_home_ptr" ]; then
        _lmstudio_home=$(cat "$_lmstudio_home_ptr")
    fi
    local _lms_installed_path="$_lmstudio_home/bin/lms"

    if [ "0" = "$NO_MODIFY_PATH" ] && [ -x "$_lms_installed_path" ]; then
        "$_lms_installed_path" bootstrap -y > /dev/null 2>&1
        local _bootstrap_result=$?
        if [ "$_bootstrap_result" -eq 0 ]; then
            local _lms_on_path
            _lms_on_path=$(command -v lms 2>/dev/null || echo "")
            if [ "$_lms_on_path" != "$_lms_installed_path" ]; then
                console_log "To add lms to your PATH, either restart your shell or run:"
                console_log "    export PATH=\"${_lmstudio_home}/bin:\$PATH\""
            fi
        else
            console_log ""
            console_log "Could not add lms to your PATH"
            console_log "lms install location: $_lms_installed_path"
        fi

        if [ -n "${GITHUB_PATH:-}" ]; then
            ensure echo "${_lmstudio_home}/bin" >> "$GITHUB_PATH"
        fi
    elif [ -x "$_lms_installed_path" ]; then
        console_log "lms install location: $_lms_installed_path"
    fi
}

download_binary_and_run_installer "$@" || exit 1

