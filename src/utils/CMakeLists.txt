# Copyright (c) 2025, Ruochun Zhang
# SPDX-License-Identifier: BSD-3-Clause

message(STATUS "==== Configuring Utility Components ====")

if(PROJECT_IS_TOP_LEVEL)
    install(
        DIRECTORY ${ProjectIncludeSource}/utils
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
        FILES_MATCHING
            PATTERN "*.h"
            PATTERN "*.hpp"
    )
endif()
