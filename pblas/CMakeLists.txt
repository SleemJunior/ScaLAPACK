cmake_minimum_required(VERSION 3.13)
enable_testing()
project(dist_mkl_exp LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 20)
find_package(MKL CONFIG REQUIRED)
find_package(MPI REQUIRED)
message(STATUS "${MKL_IMPORTED_TARGETS}") #Provides available list of targets based on input
add_executable(exp02 exp02.cc)

target_compile_options(exp02 PUBLIC $<TARGET_PROPERTY:MKL::MKL,INTERFACE_COMPILE_OPTIONS>)
target_include_directories(exp02 PUBLIC $<TARGET_PROPERTY:MKL::MKL,INTERFACE_INCLUDE_DIRECTORIES>)
#target_link_libraries(exp02 PUBLIC $<LINK_ONLY:MKL::MKL> -lmkl_blacs_intelmpi_lp64 -lmkl_scalapack_lp64)
target_link_libraries(exp02 PUBLIC  -L${MKLROOT}/lib/intel64 -lmkl_scalapack_lp64 -Wl,--no-as-needed -lmkl_intel_lp64 -lmkl_intel_thread -lmkl_core -lmkl_blacs_intelmpi_lp64 -liomp5 -lpthread -lm -ldl)
target_link_libraries(exp02 PUBLIC ${MPI_CXX_LIBRARIES})

