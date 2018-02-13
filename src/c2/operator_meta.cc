/**
 * Copyright (c) 2017-present, Facebook, Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#include "tc/c2/operator_meta.h"

namespace caffe2 {

CaffeMap<std::string, ReferenceImplementation>&
ReferenceImplementationRegistry::getMap() {
  static CaffeMap<std::string, ReferenceImplementation> instance;
  return instance;
}

void ReferenceImplementationRegistry::Append(
    NetDef* net,
    const OperatorDef& op) {
  CAFFE_ENFORCE(getMap().count(op.type()), "Not found: ", op.type());
  getMap()[op.type()](net, op);
}

NetDef ReferenceImplementationRegistry::ConvertNet(const NetDef& net) {
  NetDef result;
  result.CopyFrom(net);
  result.clear_op();
  for (const auto& op : net.op()) {
    if (getMap().count(op.type())) {
      Append(&result, op);
    } else {
      result.add_op()->CopyFrom(op);
    }
  }
  return result;
}
}
