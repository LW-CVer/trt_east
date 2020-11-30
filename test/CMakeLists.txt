add_executable(test_trt_east test.cpp)
target_link_libraries(test_trt_east ${TRT_EAST_LIBRARY_NAME})
set(OPENCV_NEED_LIBS opencv_imgproc opencv_highgui)
target_link_libraries(test_trt_east ${OPENCV_NEED_LIBS})
