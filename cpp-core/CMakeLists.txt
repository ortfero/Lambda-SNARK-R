cmake_minimum_required(VERSION 3.20)

# vcpkg integration (if available)
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/../vcpkg/installed/x64-linux")
    list(APPEND CMAKE_PREFIX_PATH "${CMAKE_CURRENT_SOURCE_DIR}/../vcpkg/installed/x64-linux")
    message(STATUS "Using vcpkg packages from: ${CMAKE_CURRENT_SOURCE_DIR}/../vcpkg/installed/x64-linux")
endif()

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

# Microsoft SEAL (FHE library)
find_package(SEAL 4.1 QUIET)
if(NOT SEAL_FOUND)
    message(STATUS "SEAL not found, will use vcpkg or manual install")
endif()

# NTL (Number Theory Library by Victor Shoup)
if(PkgConfig_FOUND)
    pkg_check_modules(NTL QUIET ntl)
endif()
if(NOT NTL_FOUND)
    find_library(NTL_LIBRARY NAMES ntl)
    find_path(NTL_INCLUDE_DIR NAMES NTL/ZZ.h)
    if(NTL_LIBRARY AND NTL_INCLUDE_DIR)
        set(NTL_FOUND TRUE)
        set(NTL_LIBRARIES ${NTL_LIBRARY})
        set(NTL_INCLUDE_DIRS ${NTL_INCLUDE_DIR})
    endif()
endif()

# GMP (GNU Multiple Precision)
find_package(GMP QUIET)
if(NOT GMP_FOUND)
    find_library(GMP_LIBRARY NAMES gmp)
    find_path(GMP_INCLUDE_DIR NAMES gmp.h)
    if(GMP_LIBRARY AND GMP_INCLUDE_DIR)
        set(GMP_FOUND TRUE)
        set(GMP_LIBRARIES ${GMP_LIBRARY})
        set(GMP_INCLUDE_DIRS ${GMP_INCLUDE_DIR})
    endif()
endif()

# Eigen3 (matrix algebra)
find_package(Eigen3 3.4 QUIET)

# libsodium (constant-time primitives, optional)
find_package(sodium QUIET)

# --- Library: lambda_snark_core ---
add_library(lambda_snark_core
    src/commitment.cpp
    src/ntt.cpp
    src/lincheck.cpp
    src/mulcheck.cpp
    src/ffi.cpp
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
if(SEAL_FOUND)
    target_link_libraries(lambda_snark_core PUBLIC SEAL::seal)
    target_compile_definitions(lambda_snark_core PRIVATE HAVE_SEAL)
    message(STATUS "SEAL found: ${SEAL_VERSION}")
else()
    message(WARNING "SEAL not found; commitment.cpp will use stub implementation")
endif()

if(NTL_FOUND)
    target_include_directories(lambda_snark_core PRIVATE ${NTL_INCLUDE_DIRS})
    target_link_libraries(lambda_snark_core PRIVATE ${NTL_LIBRARIES})
    target_compile_definitions(lambda_snark_core PRIVATE HAVE_NTL)
    message(STATUS "NTL found: ${NTL_LIBRARIES}")
else()
    message(WARNING "NTL not found; ntt.cpp will use fallback implementation")
endif()

if(GMP_FOUND)
    target_include_directories(lambda_snark_core PRIVATE ${GMP_INCLUDE_DIRS})
    target_link_libraries(lambda_snark_core PRIVATE ${GMP_LIBRARIES})
endif()

if(Eigen3_FOUND)
    target_link_libraries(lambda_snark_core PRIVATE Eigen3::Eigen)
else()
    message(WARNING "Eigen3 not found; lincheck.cpp will use basic implementation")
endif()

if(sodium_FOUND)
    target_link_libraries(lambda_snark_core PRIVATE sodium)
endif()

# Enable PIC for shared library
set_target_properties(lambda_snark_core PROPERTIES
    POSITION_INDEPENDENT_CODE ON
    VERSION ${PROJECT_VERSION}
    SOVERSION ${PROJECT_VERSION_MAJOR}
)

# --- Tests ---
if(LAMBDA_SNARK_BUILD_TESTS)
    enable_testing()
    
    # Google Test
    find_package(GTest QUIET)
    if(NOT GTest_FOUND)
        include(FetchContent)
        FetchContent_Declare(
            googletest
            GIT_REPOSITORY https://github.com/google/googletest.git
            GIT_TAG v1.14.0
        )
        FetchContent_MakeAvailable(googletest)
    endif()
    
    add_executable(test_commitment tests/test_commitment.cpp)
    target_link_libraries(test_commitment PRIVATE lambda_snark_core GTest::gtest_main)
    add_test(NAME test_commitment COMMAND test_commitment)
    
    add_executable(test_ntt tests/test_ntt.cpp)
    target_link_libraries(test_ntt PRIVATE lambda_snark_core GTest::gtest_main)
    add_test(NAME test_ntt COMMAND test_ntt)
endif()

# --- Benchmarks ---
if(LAMBDA_SNARK_BUILD_BENCHMARKS)
    find_package(benchmark QUIET)
    if(benchmark_FOUND)
        add_executable(bench_ntt benches/bench_ntt.cpp)
        target_link_libraries(bench_ntt PRIVATE lambda_snark_core benchmark::benchmark)
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
message(STATUS "  NTL found:    ${NTL_FOUND}")
message(STATUS "  GMP found:    ${GMP_FOUND}")
message(STATUS "  Eigen3 found: ${Eigen3_FOUND}")
message(STATUS "  Sodium found: ${sodium_FOUND}")
message(STATUS "  Tests:        ${LAMBDA_SNARK_BUILD_TESTS}")
message(STATUS "  Benchmarks:   ${LAMBDA_SNARK_BUILD_BENCHMARKS}")
message(STATUS "  Native arch:  ${LAMBDA_SNARK_NATIVE_ARCH}")
message(STATUS "")
