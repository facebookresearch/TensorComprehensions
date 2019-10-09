/**
 * Copyright (c) 2018, Facebook, Inc.
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
#include "tc/core/polyhedral/utils.h"

#include "tc/core/polyhedral/domain_types.h"

namespace tc {
namespace polyhedral {

/* Construct a tuple representing the tensor with identifier "tensorId" and
 * dimension "dim" from the parameter space "paramSpace",
 * without any specific names for the indices, from the perspective
 * of the user.
 * Since some names are required, use names of the form "__tc_tensor_arg*".
 */
isl::MultiId<Tensor>
constructTensorTuple(isl::Space<> paramSpace, isl::id tensorId, size_t dim) {
  auto tensorSpace = paramSpace.add_named_tuple_id_ui<Tensor>(tensorId, dim);
  isl::id_list tensorArgs(paramSpace.get_ctx(), 0);
  for (size_t i = 0; i < dim; ++i) {
    auto name = std::string("__tc_tensor_arg") + std::to_string(i);
    tensorArgs = tensorArgs.add(isl::id(paramSpace.get_ctx(), name));
  }
  return isl::MultiId<Tensor>(tensorSpace, tensorArgs);
}

} // namespace polyhedral
} // namespace tc
