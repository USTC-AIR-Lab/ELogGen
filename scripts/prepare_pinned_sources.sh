#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ -f "$PROJECT_ROOT/.eloggen.local" ]]; then
    source "$PROJECT_ROOT/.eloggen.local"
fi

DATA_ROOT="${ELOGGEN_DATA_ROOT:-$PROJECT_ROOT}"
BEHAVIOR_REPOSITORY="${ELOGGEN_BEHAVIOR_REPOSITORY:-https://github.com/StanfordVL/BEHAVIOR-1K.git}"
ROBOMIMIC_REPOSITORY="${ELOGGEN_ROBOMIMIC_REPOSITORY:-https://github.com/ChengshuLi/robomimic.git}"
GIT_PROXY="${ELOGGEN_GIT_PROXY:-}"
DRY_RUN=false

while [[ $# -gt 0 ]]; do
    case "$1" in
        --data-root) DATA_ROOT="$2"; shift 2 ;;
        --behavior-repository) BEHAVIOR_REPOSITORY="$2"; shift 2 ;;
        --robomimic-repository) ROBOMIMIC_REPOSITORY="$2"; shift 2 ;;
        --git-proxy) GIT_PROXY="$2"; shift 2 ;;
        --dry-run|-h|--help) DRY_RUN=true; shift ;;
        --skip-tests|--skip-environment-capture|--skip-lock-capture|--prepare-only) shift ;;
        *) shift ;;
    esac
done

if [[ "$DRY_RUN" == true ]]; then
    exit 0
fi

THIRD_PARTY_ROOT="$DATA_ROOT/third_party"
SOURCE_CACHE_ROOT="$DATA_ROOT/cache/sources"
BEHAVIOR_ROOT="$THIRD_PARTY_ROOT/BEHAVIOR-1K"
ROBOMIMIC_ROOT="$THIRD_PARTY_ROOT/robomimic"
EXPECTED_COMMIT="$(awk '/^commit:/ {print $2; exit}' "$PROJECT_ROOT/src/eloggen/patches/behavior1k/version.yaml")"
EXPECTED_ROBOMIMIC_COMMIT="$(awk '/^commit:/ {print $2; exit}' "$PROJECT_ROOT/src/eloggen/patches/robomimic/version.yaml")"

mkdir -p "$THIRD_PARTY_ROOT" "$SOURCE_CACHE_ROOT"

CURL_OPTIONS=(--fail --location --retry 10 --retry-all-errors --retry-delay 2 --connect-timeout 30)
GIT_OPTIONS=(-c http.version=HTTP/1.1)
if [[ -n "$GIT_PROXY" ]]; then
    CURL_OPTIONS+=(--proxy "$GIT_PROXY")
    GIT_OPTIONS+=(-c "http.proxy=$GIT_PROXY")
fi

github_codeload_url() {
    local repository="$1"
    local commit="$2"
    local path
    if [[ "$repository" != https://github.com/* ]]; then
        return 1
    fi
    path="${repository#https://github.com/}"
    path="${path%.git}"
    if [[ "$path" != */* ]]; then
        return 1
    fi
    printf 'https://codeload.github.com/%s/tar.gz/%s\n' "$path" "$commit"
}

prepare_archive_source() {
    local label="$1"
    local repository="$2"
    local commit="$3"
    local destination="$4"
    local cache_name="$5"
    local archive_url
    local archive
    local archive_part
    local work_root
    local source_root
    local metadata_ok=false

    if [[ -d "$destination/.git" ]]; then
        return 0
    fi
    if [[ -e "$destination" ]]; then
        return 0
    fi
    if ! archive_url="$(github_codeload_url "$repository" "$commit")"; then
        echo "[source] $label repository is not a github.com HTTPS URL; using Git fetch fallback."
        return 0
    fi

    archive="$SOURCE_CACHE_ROOT/${cache_name}-${commit}.tar.gz"
    archive_part="$archive.part"
    if [[ ! -f "$archive" ]]; then
        echo "Downloading pinned $label archive from codeload.github.com..."
        rm -f -- "$archive_part"
        curl "${CURL_OPTIONS[@]}" --output "$archive_part" "$archive_url"
        mv "$archive_part" "$archive"
    else
        echo "Using cached pinned $label archive: $archive"
    fi

    work_root="$(mktemp -d "$THIRD_PARTY_ROOT/.${cache_name}-archive.XXXXXX")"
    source_root="$work_root/source"
    mkdir -p "$source_root"
    cleanup_archive_source() {
        rm -f -- "$archive_part"
        if [[ -n "${work_root:-}" && -d "$work_root" ]]; then
            rm -rf -- "$work_root"
        fi
    }
    trap cleanup_archive_source EXIT

    echo "Extracting pinned $label archive..."
    tar -xzf "$archive" --strip-components=1 -C "$source_root"

    # Attach only commit / tree metadata. File contents came from the much
    # faster codeload archive, so this fetch deliberately omits blobs.
    git init -q "$source_root"
    git -C "$source_root" remote add origin "$repository"
    for attempt in 1 2 3; do
        echo "Fetching lightweight $label Git metadata (attempt $attempt/3)..."
        if git "${GIT_OPTIONS[@]}" -C "$source_root" fetch \
            --depth=1 --filter=blob:none --no-tags origin "$commit"; then
            metadata_ok=true
            break
        fi
        sleep $((attempt * 2))
    done
    if [[ "$metadata_ok" != true ]]; then
        echo "Failed to fetch lightweight $label Git metadata after three attempts." >&2
        return 1
    fi

    # Point HEAD / index at the pinned commit without checking files out again.
    # A checkout would defeat the archive fast path by lazy-fetching all blobs.
    git -C "$source_root" reset --mixed --quiet FETCH_HEAD
    if [[ "$(git -C "$source_root" rev-parse HEAD)" != "$commit" ]]; then
        echo "$label metadata commit mismatch after archive extraction." >&2
        return 1
    fi
    printf '%s\n' "$commit" > "$source_root/.eloggen-source-commit"

    # apply_patches.py reads this one upstream file with `git show`; seed its
    # blob locally so that verification does not trigger another network fetch.
    if [[ "$label" == "robomimic" && -f "$source_root/robomimic/envs/env_omnigibson.py" ]]; then
        git -C "$source_root" hash-object -w \
            "$source_root/robomimic/envs/env_omnigibson.py" >/dev/null
    fi

    mv "$source_root" "$destination"
    rmdir "$work_root"
    work_root=""
    trap - EXIT
    echo "[OK] prepared pinned $label source at $destination"
}

prepare_archive_source \
    "BEHAVIOR-1K" \
    "$BEHAVIOR_REPOSITORY" \
    "$EXPECTED_COMMIT" \
    "$BEHAVIOR_ROOT" \
    "behavior"

prepare_archive_source \
    "robomimic" \
    "$ROBOMIMIC_REPOSITORY" \
    "$EXPECTED_ROBOMIMIC_COMMIT" \
    "$ROBOMIMIC_ROOT" \
    "robomimic"
