#!/bin/bash
if [ "$#" -ne 1 ]; then
    echo "Please specify directory containing Spider files."
    exit 1
fi

realpath() {
  OURPWD=$PWD
  cd "$(dirname "$1")"
  LINK=$(readlink "$(basename "$1")")
  while [ "$LINK" ]; do
    cd "$(dirname "$LINK")"
    LINK=$(readlink "$(basename "$1")")
  done
  REALPATH="$PWD/$(basename "$1")"
  cd "$OURPWD"
  echo "$REALPATH"
}
realpath "$@"

BASE=$(realpath $(dirname $0))

# Re-generate 'sql' to fix bad parsing
cp $1/tables.json ${BASE}
for input in train_others train_spider dev; do
    echo Procesing $input
    cp $1/${input}.json ${BASE}
    if [[ -e ${BASE}/${input}.json.patch ]]; then
        pushd ${BASE} >& /dev/null
        patch < ${input}.json.patch
        popd >& /dev/null
    fi
        python -m seq2struct.datasets.spider_lib.preprocess.parse_raw_json \
        --tables ${BASE}/tables.json \
        --input ${BASE}/${input}.json \
        --output ${BASE}/${input}.json
    echo
done
