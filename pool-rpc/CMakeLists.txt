# Added to stable-diffusion.cpp's examples by the build workflow.
add_executable(pool-rpc-server rpc-server.cpp)
target_link_libraries(pool-rpc-server PRIVATE ggml)
target_compile_features(pool-rpc-server PRIVATE cxx_std_17)
install(TARGETS pool-rpc-server RUNTIME)
