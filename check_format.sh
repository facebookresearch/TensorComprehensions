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
    echo ${f} is not properly formatted.
    MUST_FORMAT=1
    ${CLANG} -style=file ${f} > /tmp/$(basename $f)
    diff -y $f /tmp/$(basename $f) -W 180    
    diff -y $f /tmp/$(basename $f) --suppress-common-lines -W 180    
    break
  fi
done
exit ${MUST_FORMAT}
