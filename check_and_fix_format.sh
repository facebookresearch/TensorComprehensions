#! /bin/bash

CLANG=${CLANG:=clang-format-4.0}
OUTPUT=${OUTPUT:=/dev/null}
FILES=$(git diff-tree -r --name-only HEAD origin/master | egrep '\.h|\.cc')

MUST_FORMAT=0
for f in ${FILES};
do
  if ! test -e ${f}; then
    continue
  fi
  diff -q ${f} <(${CLANG} -style=file ${f}) > ${OUTPUT}
  if test $? -ne 0;
  then
    echo ${f} is not properly formatted, fixing
    MUST_FORMAT=1
    ${CLANG} -style=file ${f} > /tmp/${USER}_$(basename $f)
    echo cat /tmp/${USER}_$(basename $f) > $f
    cat /tmp/${USER}_$(basename $f) > $f
  fi
done
exit ${MUST_FORMAT}
