cmake_minimum_required(VERSION 3.20)

# vcpkg toolchain (honor existing CMAKE_TOOLCHAIN_FILE, otherwise pick from VCPKG_ROOT)
if(NOT DEFINED CMAKE_TOOLCHAIN_FILE AND DEFINED ENV{VCPKG_ROOT})
    set(_VCPKG_TOOLCHAIN "$ENV{VCPKG_ROOT}/scripts/buildsystems/vcpkg.cmake")
    if(EXISTS "${_VCPKG_TOOLCHAIN}")
        set(CMAKE_TOOLCHAIN_FILE "${_VCPKG_TOOLCHAIN}" CACHE STRING "Vcpkg toolchain file")
        message(STATUS "Using vcpkg toolchain: ${CMAKE_TOOLCHAIN_FILE}")
    endif()
endif()

# vcpkg integration (local layout, useful when toolchain not passed explicitly)
set(_VCPKG_LOCAL_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../vcpkg")
foreach(_triplet IN ITEMS x64-linux x64-osx x64-windows)
    set(_triplet_path "${_VCPKG_LOCAL_ROOT}/installed/${_triplet}")
    if(EXISTS "${_triplet_path}")
        list(APPEND CMAKE_PREFIX_PATH "${_triplet_path}")
        message(STATUS "Using vcpkg packages from: ${_triplet_path}")
    endif()
endforeach()
unset(_triplet_path)
unset(_triplet)
unset(_VCPKG_LOCAL_ROOT)
unset(_VCPKG_TOOLCHAIN)

project(LambdaSNARK_Core
    VERSION 0.1.0
    DESCRIPTION "ΛSNARK-R C++ Performance Kernel"
    LANGUAGES CXX C
)

# --- Options ---
option(LAMBDA_SNARK_BUILD_TESTS "Build C++ tests" ON)
option(LAMBDA_SNARK_BUILD_BENCHMARKS "Build benchmarks" ON)
option(LAMBDA_SNARK_USE_ASAN "Enable AddressSanitizer" OFF)
option(LAMBDA_SNARK_USE_UBSAN "Enable UndefinedBehaviorSanitizer" OFF)
option(LAMBDA_SNARK_NATIVE_ARCH "Compile for native CPU (-march=native)" ON)

# --- C++ Standard ---
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# --- Compiler Flags ---
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    add_compile_options(
        -Wall -Wextra -Wpedantic
        -Wconversion -Wsign-conversion
        -Wno-unused-parameter
    )
    
    if(LAMBDA_SNARK_NATIVE_ARCH)
        add_compile_options(-march=native)
    endif()
    
    # Release optimizations
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3 -DNDEBUG")
    
    # Sanitizers
    if(LAMBDA_SNARK_USE_ASAN)
        add_compile_options(-fsanitize=address -fno-omit-frame-pointer)
        add_link_options(-fsanitize=address)
    endif()
    
    if(LAMBDA_SNARK_USE_UBSAN)
        add_compile_options(-fsanitize=undefined)
        add_link_options(-fsanitize=undefined)
    endif()
endif()

# --- Dependencies (via vcpkg or system) ---
find_package(PkgConfig QUIET)

# Microsoft SEAL (FHE library) - required
find_package(SEAL 4.1 REQUIRED CONFIG)

# GMP (required) via pkg-config (vcpkg provides .pc)
pkg_check_modules(GMP REQUIRED IMPORTED_TARGET gmp)
set(GMP_FOUND TRUE)

# Eigen3 (matrix algebra)
find_package(Eigen3 3.4 QUIET)

# libsodium (constant-time primitives, optional)
find_package(sodium QUIET)

# NTL (number theory library) - required
find_package(NTL QUIET)
if(NOT NTL_FOUND)
    find_library(NTL_LIBRARY NAMES ntl)
    find_path(NTL_INCLUDE_DIR NAMES NTL/ZZ_p.h)
    if(NTL_LIBRARY AND NTL_INCLUDE_DIR)
        add_library(NTL::ntl UNKNOWN IMPORTED)
        set_target_properties(NTL::ntl PROPERTIES
            IMPORTED_LOCATION ${NTL_LIBRARY}
            INTERFACE_INCLUDE_DIRECTORIES ${NTL_INCLUDE_DIR}
        )
        set(NTL_FOUND TRUE)
    endif()
endif()
if(NOT NTL_FOUND)
    message(FATAL_ERROR "NTL not found; install NTL and ensure headers/libs are discoverable (e.g., via vcpkg or system package).")
endif()

# --- Library: lambda_snark_core ---
# --- Main library ---
add_library(lambda_snark_core STATIC
    src/commitment.cpp
    src/ntt.cpp
    src/lincheck.cpp
    src/mulcheck.cpp
    src/r1cs.cpp
    src/ffi.cpp
    src/lean_ffi.cpp
    src/utils.cpp
)

target_include_directories(lambda_snark_core
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
)

# Link dependencies
target_link_libraries(lambda_snark_core PUBLIC SEAL::seal)
target_compile_definitions(lambda_snark_core PUBLIC HAVE_SEAL LAMBDA_SNARK_HAS_SEAL=1)
message(STATUS "SEAL found: ${SEAL_VERSION}")

if(GMP_FOUND)
    target_include_directories(lambda_snark_core PRIVATE ${GMP_INCLUDE_DIRS})
    if(GMP_LIBRARY_DIRS)
        target_link_directories(lambda_snark_core PUBLIC $<BUILD_INTERFACE:${GMP_LIBRARY_DIRS}>)
    endif()
    # Prefer pkg-config imported target for proper include/lib paths
    if(TARGET PkgConfig::gmp)
        target_link_libraries(lambda_snark_core PUBLIC PkgConfig::gmp)
    else()
        target_link_libraries(lambda_snark_core PUBLIC ${GMP_LIBRARIES})
    endif()
endif()

if(Eigen3_FOUND)
    target_link_libraries(lambda_snark_core PRIVATE Eigen3::Eigen)
else()
    message(WARNING "Eigen3 not found; lincheck.cpp will use basic implementation")
endif()

if(sodium_FOUND)
    target_link_libraries(lambda_snark_core PRIVATE sodium)
endif()

if(NTL_FOUND)
    target_link_libraries(lambda_snark_core PUBLIC NTL::ntl)
else()
    message(WARNING "NTL not found; R1CS functionality will be unavailable")
endif()

# Enable PIC for shared library
set_target_properties(lambda_snark_core PROPERTIES
    POSITION_INDEPENDENT_CODE ON
    VERSION ${PROJECT_VERSION}
    SOVERSION ${PROJECT_VERSION_MAJOR}
)

option(LAMBDA_SNARK_DOWNLOAD_GTEST "Download GTest with FetchContent when not found" OFF)

# --- Tests ---
if(LAMBDA_SNARK_BUILD_TESTS)
    enable_testing()
    find_package(GTest REQUIRED)

    add_executable(test_commitment tests/test_commitment.cpp)
    target_link_libraries(test_commitment PRIVATE lambda_snark_core GTest::gtest_main)
    add_test(NAME test_commitment COMMAND test_commitment)

    add_executable(test_ntt tests/test_ntt.cpp)
    target_link_libraries(test_ntt PRIVATE lambda_snark_core GTest::gtest_main)
    add_test(NAME test_ntt COMMAND test_ntt)

    add_executable(test_r1cs tests/test_r1cs.cpp)
    target_link_libraries(test_r1cs PRIVATE lambda_snark_core GTest::gtest_main)
    add_test(NAME test_r1cs COMMAND test_r1cs)

    add_executable(test_conformance tests/test_conformance.cpp)
    target_link_libraries(test_conformance PRIVATE lambda_snark_core GTest::gtest_main)
    add_test(NAME test_conformance COMMAND test_conformance)

    add_executable(test_utils tests/test_utils.cpp)
    target_link_libraries(test_utils PRIVATE lambda_snark_core GTest::gtest_main)
    add_test(NAME test_utils COMMAND test_utils)
endif()

# --- Tooling ---
add_executable(dudect_sampler tools/dudect_sampler.cpp)
target_link_libraries(dudect_sampler PRIVATE lambda_snark_core)

# --- Benchmarks ---
if(LAMBDA_SNARK_BUILD_BENCHMARKS)
    find_package(benchmark QUIET)
    if(benchmark_FOUND)
        # TODO: Create benches/bench_ntt.cpp
        # add_executable(bench_ntt benches/bench_ntt.cpp)
        # target_link_libraries(bench_ntt PRIVATE lambda_snark_core benchmark::benchmark)
        message(STATUS "Benchmarks enabled but benches/ directory not yet created")
    else()
        message(STATUS "Google Benchmark not found; skipping benchmarks")
    endif()
endif()

# --- Install ---
include(GNUInstallDirs)

install(TARGETS lambda_snark_core
    EXPORT LambdaSNARKCoreTargets
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)

install(DIRECTORY include/
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

install(EXPORT LambdaSNARKCoreTargets
    FILE LambdaSNARKCoreTargets.cmake
    NAMESPACE LambdaSNARK::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/LambdaSNARKCore
)

# --- Package Config ---
include(CMakePackageConfigHelpers)

configure_package_config_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/cmake/LambdaSNARKCoreConfig.cmake.in
    ${CMAKE_CURRENT_BINARY_DIR}/LambdaSNARKCoreConfig.cmake
    INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/LambdaSNARKCore
)

write_basic_package_version_file(
    ${CMAKE_CURRENT_BINARY_DIR}/LambdaSNARKCoreConfigVersion.cmake
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY SameMajorVersion
)

install(FILES
    ${CMAKE_CURRENT_BINARY_DIR}/LambdaSNARKCoreConfig.cmake
    ${CMAKE_CURRENT_BINARY_DIR}/LambdaSNARKCoreConfigVersion.cmake
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/LambdaSNARKCore
)

# --- Summary ---
message(STATUS "")
message(STATUS "ΛSNARK-R C++ Core Configuration:")
message(STATUS "  Version:      ${PROJECT_VERSION}")
message(STATUS "  Build type:   ${CMAKE_BUILD_TYPE}")
message(STATUS "  C++ standard: ${CMAKE_CXX_STANDARD}")
message(STATUS "  SEAL found:   ${SEAL_FOUND}")
message(STATUS "  GMP found:    ${GMP_FOUND}")
message(STATUS "  Eigen3 found: ${Eigen3_FOUND}")
message(STATUS "  Sodium found: ${sodium_FOUND}")
message(STATUS "  NTL found:    ${NTL_FOUND}")
message(STATUS "  Tests:        ${LAMBDA_SNARK_BUILD_TESTS}")
message(STATUS "  Benchmarks:   ${LAMBDA_SNARK_BUILD_BENCHMARKS}")
message(STATUS "  Native arch:  ${LAMBDA_SNARK_NATIVE_ARCH}")
message(STATUS "")
