#!/bin/bash
# A fake rfpii for the API's tests. It answers `--version`, `inventory
# DIR --json`, `job JOB.yaml` and `verify PDF --manifest M ...` as the real
# binary does, as fake.conf (beside it) says:
#   FORMS     forms the inventory counts (one photo each)
#   RELEASED  forms the job releases (the rest are held)
#   DELAY     seconds the job takes
#   EXIT      the job's exit code ("-": 0 if all were released, else 2)
#   TOKENS_IN, TOKENS_OUT   tokens the report says gemma4:31b used
#   NO_REPORT 1: the job writes no report
DIR="$(cd "$(dirname "$0")" && pwd)"
FORMS=3
RELEASED=3
DELAY=0
EXIT=-
TOKENS_IN=1000
TOKENS_OUT=200
NO_REPORT=0
[ -f "$DIR/fake.conf" ] && . "$DIR/fake.conf"

inventory() {
    printf '{"schema":"rfpii/inventory/1","totals":{"files":%d,"pages":%d,"duplicates":0,"broken":0,"repaired":0},"files":[' "$FORMS" "$FORMS"
    for i in $(seq 1 "$FORMS"); do
        [ "$i" -gt 1 ] && printf ','
        printf '{"path":"f%d.jpg","kind":"photo","pages":1,"sha256":"%064d","group":"photo"}' "$i" "$i"
    done
    printf ']}\n'
}

job() {
    local job_file="$1"
    local out
    out=$(sed -n 's/^outputs: *//p' "$job_file")
    echo $$ > "$DIR/job.pid"
    cp "$job_file" "$DIR/last-job.yaml"
    cp "$(dirname "$job_file")/controller.json" "$DIR/last-controller.json"
    ls "$(dirname "$job_file")/in" > "$DIR/last-inputs.txt"
    echo "fake rfpii job: $job_file"
    sleep "$DELAY"
    mkdir -p "$out/public" "$out/staging" "$out/review"
    echo '{"id":"t1","recipient":{"host":"ollama.com","region":"foreign"}}' > "$out/staging/transfers.jsonl"
    echo '{"document":"held.jpg","reason":"no template matched"}' > "$out/review/held.json"
    if [ "$RELEASED" -gt 0 ]; then
        mkdir -p "$out/public/batches"
        for i in $(seq 1 "$RELEASED"); do
            echo "%PDF-1.4 fake form $i" > "$out/public/form$i.pdf"
            echo "{\"form\": $i}" > "$out/public/form$i.manifest.json"
        done
        echo '{"entry": 0}' > "$out/public/ledger.jsonl"
        echo 'PUBLIC KEY' > "$out/public/signing_key.pub.pem"
        cp "$out/staging/transfers.jsonl" "$out/public/transfers.jsonl"
        echo '{"merkle_root":"ab"}' > "$out/public/batches/batch-1.json"
    fi
    local held=$((FORMS - RELEASED))
    if [ "$NO_REPORT" != 1 ]; then
        cat > "$out/report.json" <<EOF
{"schema":"rfpii/job-report/1","plan":{"documents":$FORMS},"groups":[{"id":"photo","documents":$FORMS,"published":$FORMS,"held":$held,"released":$RELEASED}],"held":{"documents":$held},"model_usage":[{"host":"ollama.com","model":"gemma4:31b","requests":4,"input_tokens":$TOKENS_IN,"output_tokens":$TOKENS_OUT}]}
EOF
        echo "# Redaction job: report" > "$out/report.md"
    fi
    if [ "$EXIT" = "-" ]; then
        [ "$held" -gt 0 ] && exit 2
        exit 0
    fi
    echo "error: the fake was told to fail" >&2
    exit "$EXIT"
}

verify() {
    # verify PDF --manifest M [...]
    local manifest="$4"
    echo "$@" > "$DIR/last-verify.txt"
    if grep -q '"ok"' "$manifest"; then
        echo "PASS  manifest signature valid"
        echo "PASS  redacted PDF matches the manifest's SHA-256"
        echo "note  the PDF is not digitally signed"
        echo "RESULT: VERIFIED - only the listed blocks were blacked out; nothing else changed"
        exit 0
    fi
    echo "PASS  manifest signature valid"
    echo "FAIL  redacted PDF matches the manifest's SHA-256"
    echo "RESULT: NOT VERIFIED"
    exit 1
}

case "$1" in
    --version) echo "rfpii 0.3.0 (fake)" ;;
    inventory) inventory ;;
    job) job "$2" ;;
    verify) verify "$@" ;;
    *) echo "fake rfpii: unknown command $1" >&2; exit 1 ;;
esac
