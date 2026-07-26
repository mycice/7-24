# Best-effort deploy of the built DLL into the Unity plugins folder (invoked as a POST_BUILD step).
# NON-FATAL by design: while the Unity Editor is open it holds a lock on the DLL and the copy fails;
# RESULT_VARIABLE captures that so the build still succeeds, and the user does the usual
# close-Unity-then-copy. When Unity is closed, the build auto-deploys so source and shipped binary
# can never silently diverge (review wroqdtb9s).
execute_process(
    COMMAND ${CMAKE_COMMAND} -E copy_if_different "${SRC}" "${DST}"
    RESULT_VARIABLE _copy_result
    ERROR_QUIET
)
if(_copy_result EQUAL 0)
    message(STATUS "[deploy] LiverCudaSim.dll -> Assets/Plugins/x86_64 (auto)")
else()
    message(STATUS "[deploy] skipped — Unity likely holds the lock; close Unity and copy "
                   "build/Release/LiverCudaSim.dll -> Assets/Plugins/x86_64/ manually")
endif()
