aux_source_directory(./ trteast_src)
add_library(${TRT_EAST_LIBRARY_NAME} SHARED ${trteast_src})

set(OPENCV_NEED_LIBS opencv_imgproc)
target_link_libraries(${TRT_EAST_LIBRARY_NAME} PRIVATE ${OPENCV_NEED_LIBS} )
target_link_libraries(${TRT_EAST_LIBRARY_NAME} PRIVATE ${CUDA_LIBS})
target_link_libraries(${TRT_EAST_LIBRARY_NAME} PRIVATE ${TRT_LIBS})