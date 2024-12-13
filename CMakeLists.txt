project(par_for_perf LANGUAGES CXX)

add_subdirectory(external/parthenon)

add_executable(
   par_for_perf
   main.cpp
   )

target_link_libraries(par_for_perf PRIVATE Parthenon::parthenon)


