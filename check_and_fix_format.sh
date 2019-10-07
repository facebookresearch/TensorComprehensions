# Copyright (c) 2017-present, Facebook, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
##############################################################################
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
