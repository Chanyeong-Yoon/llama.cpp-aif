set(TARGET llama-aif-report)

add_executable(${TARGET} aif-report.cpp)
target_compile_features(${TARGET} PRIVATE cxx_std_17)

if(LLAMA_TOOLS_INSTALL)
    install(TARGETS ${TARGET} RUNTIME)
endif()
