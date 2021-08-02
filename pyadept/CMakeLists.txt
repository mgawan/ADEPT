#
# PyADEPT
#
set (CMAKE_INSTALL_PYTHONDIR ${CMAKE_INSTALL_LIBDIR})

set(_PYLIB ${CMAKE_INSTALL_PYTHONDIR}/${PROJECT_NAME})

# no submodules
set(PYTHON_SUBMODULE_FOLDERS options)

file(GLOB PYTHON_SUBMODULE_FILES ${PROJECT_SOURCE_DIR}/${PROJECT_NAME}/*.py)
string(REPLACE "${PROJECT_SOURCE_DIR}/${PROJECT_NAME}/" ""
    PYTHON_SUBMODULE_FILES "${PYTHON_SUBMODULE_FILES}")

foreach(_SUBMODULE ${PYTHON_SUBMODULE_FOLDERS})
    set(SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/${_SUBMODULE})
    file(GLOB_RECURSE _FILES ${SOURCE_DIR}/*.py ${SOURCE_DIR}/*.py.in)

    # copy over the python file
    foreach(_FILE ${_FILES})
        get_filename_component(_DIR "${_FILE}" DIRECTORY)
        get_filename_component(_FILE "${_FILE}" NAME_WE)
        string(REPLACE "${PROJECT_SOURCE_DIR}/" "" _DIR "${_DIR}")
        if(EXISTS ${PROJECT_SOURCE_DIR}/${_DIR}/${_FILE}.py)
            configure_file(${PROJECT_SOURCE_DIR}/${_DIR}/${_FILE}.py
                ${PROJECT_BINARY_DIR}/${_DIR}/${_FILE}.py @ONLY)
        elseif(EXISTS ${PROJECT_SOURCE_DIR}/${_DIR}/${_FILE}.py.in)
            configure_file(${PROJECT_SOURCE_DIR}/${_DIR}/${_FILE}.py.in
                ${PROJECT_BINARY_DIR}/${_DIR}/${_FILE}.py @ONLY)
        endif()
        if(CMAKE_INSTALL_PYTHONDIR)
            install(FILES ${PROJECT_BINARY_DIR}/${_DIR}/${_FILE}.py
                DESTINATION ${CMAKE_INSTALL_PYTHONDIR}/${_DIR}
                OPTIONAL)
        endif()
    endforeach()
endforeach()

#----------------------------------------------------------------------------
# Install the python stuff
#
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/__init__.py.in
    ${CMAKE_CURRENT_BINARY_DIR}/__init__.py @ONLY)

if(CMAKE_INSTALL_PYTHONDIR)
    install(FILES ${CMAKE_CURRENT_BINARY_DIR}/__init__.py
        DESTINATION ${CMAKE_INSTALL_PYTHONDIR}/py${PROJECT_NAME}
        OPTIONAL)
endif()

foreach(PYLIB_SUBMODULE ${PYTHON_SUBMODULE_FILES})
    get_filename_component(PYLIB_SUBMODULE "${PYLIB_SUBMODULE}" NAME_WE)
    configure_file(${CMAKE_CURRENT_SOURCE_DIR}/${PYLIB_SUBMODULE}.py
        ${CMAKE_CURRENT_BINARY_DIR}/${PYLIB_SUBMODULE}.py @ONLY)

    if(CMAKE_INSTALL_PYTHONDIR)
        install(FILES ${CMAKE_CURRENT_BINARY_DIR}/${PYLIB_SUBMODULE}.py
            DESTINATION ${CMAKE_INSTALL_PYTHONDIR}/py${PROJECT_NAME}
            OPTIONAL)
    endif()
endforeach()
