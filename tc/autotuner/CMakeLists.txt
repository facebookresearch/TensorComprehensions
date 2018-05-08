set(AUTOTUNER_FILES
  genetic_search.cc
  parameters.cc
  utils.cc
  cpu/autotuner.cc
)

set(TC_LIBRARIES ${TC_LIBRARIES} tc_core_cpu)

if (WITH_CUDA)
set(AUTOTUNER_FILES ${AUTOTUNER_FILES} cuda/autotuner.cc)
set(TC_LIBRARIES ${TC_LIBRARIES} tc_cuda)
endif()

add_library(
  tc_autotuner
  SHARED
  ${AUTOTUNER_FILES}
)

target_include_directories(tc_autotuner PUBLIC ${PROJECT_SOURCE_DIR}/include)
target_link_libraries(tc_autotuner PUBLIC ${ATEN_LIBRARIES} ${TC_LIBRARIES})

install(TARGETS tc_autotuner DESTINATION lib)
