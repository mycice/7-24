# LiverCudaSim  - native CUDA Unity plugin.
# Stage 1: preprocess marshal + DC reconstruction (static).
# Build (from this folder, with CUDA 12.x + VS on PATH):
#   cmake -B build -A x64
#   cmake --build build --config Release
# Output: build/Release/LiverCudaSim.dll  ->  copy to  <UnityProject>/Assets/Plugins/x86_64/
cmake_minimum_required(VERSION 3.18)
project(LiverCudaSim LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# plugin_api.cpp includes the shared CUDA-facing headers.  Link cudart through
# CMake's toolkit target so the C++ compiler receives the CUDA include path too.
find_package(CUDAToolkit REQUIRED)

add_library(LiverCudaSim SHARED
    src/plugin_api.cpp      # exported C ABI (host C++)
    cuda/dc_recon.cu        # DC reconstruction kernels + host API (CUDA)
    cuda/physics.cu         # physics host API  - body swapped to XPBD (design rev-B 9-(4) GO-LIVE)
    cuda/cut.cu             # cutting: paper Eq6-8 M-T detect + sever + cut-FP chain (CUDA)
    cuda/xpbd/xpbd_solver.cu # XPBD solver module  - bound to the shared buffers by physics_init (9-(4))
)

# cuda/ holds common.cuh + dc_recon.h, included by both translation units.
target_include_directories(LiverCudaSim PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/cuda)
target_link_libraries(LiverCudaSim PRIVATE CUDA::cudart)

set_target_properties(LiverCudaSim PROPERTIES
    CUDA_ARCHITECTURES "86"            # RTX 3090 = sm_86
    CUDA_SEPARABLE_COMPILATION ON
    POSITION_INDEPENDENT_CODE ON
)
# Stage 1 links only the CUDA runtime (provided automatically by the CUDA language).
# Stage 4 will add: target_link_libraries(LiverCudaSim PRIVATE d3d11) + the Unity PluginAPI headers.

# Best-effort auto-deploy into the Unity plugins folder so source and shipped binary can't silently
# diverge (review wroqdtb9s). NON-FATAL: skips cleanly if the Unity Editor holds the DLL lock.
add_custom_command(TARGET LiverCudaSim POST_BUILD
    COMMAND ${CMAKE_COMMAND}
            -DSRC=$<TARGET_FILE:LiverCudaSim>
            -DDST=${CMAKE_CURRENT_SOURCE_DIR}/../Assets/Plugins/x86_64/LiverCudaSim.dll
            -P ${CMAKE_CURRENT_SOURCE_DIR}/deploy.cmake
    VERBATIM
)

# ── XPBD module (round-3 design rev-B)  - standalone self-test executable. ─────────────────────────
# The module lives under cuda/xpbd/ and (since migration step 4) is ALSO linked into LiverCudaSim.dll
# above  - physics_init binds it to the shared cornerPos/vel buffers and physics_step runs its substep
# loop. This target keeps the module's PRIVATE-buffer self-test path (xpbd_init) exercisable.
# Build+run:  cmake --build build --config Release --target xpbd_selftest
#             build/Release/xpbd_selftest.exe   (exit 0 = all acceptance checks pass)
add_executable(xpbd_selftest
    tests/xpbd_selftest.cu
    cuda/xpbd/xpbd_solver.cu
)
set_target_properties(xpbd_selftest PROPERTIES
    CUDA_ARCHITECTURES "86"
    CUDA_SEPARABLE_COMPILATION ON
)

# ── XPBD LIVE-PATH gates (migration step 4 VERIFY, design rev-B 9-(4))  - headless test executable. ─
# Links the REAL pipeline sources (recon + physics + cut + xpbd; no plugin_api needed) and
# drives physics_init/physics_step on a synthetic grid: divergence check (recon_readback shows the
# XPBD deformation), golden transfer (bound path == standalone module), sever-propagation (the exact
# k_SeverLinks write pattern on the LIVE nbrIdx drops the constraint), debris-freeze, multi-sever.
# Build+run:  cmake --build build --config Release --target xpbd_livepath_test
add_executable(xpbd_livepath_test
    tests/xpbd_livepath_test.cu
    cuda/dc_recon.cu
    cuda/physics.cu
    cuda/cut.cu
    cuda/xpbd/xpbd_solver.cu
)
target_include_directories(xpbd_livepath_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/cuda)
set_target_properties(xpbd_livepath_test PROPERTIES
    CUDA_ARCHITECTURES "86"
    CUDA_SEPARABLE_COMPILATION ON
)
