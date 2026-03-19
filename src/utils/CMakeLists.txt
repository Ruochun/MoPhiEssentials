# Copyright (c) 2025, Ruochun Zhang
# SPDX-License-Identifier: BSD-3-Clause

message(STATUS "==== Configuring Utility Components ====")

if(MPE_ENABLE_INSTALL)
    install(
        DIRECTORY ${ProjectIncludeSource}/utils
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
        FILES_MATCHING
            PATTERN "*.h"
            PATTERN "*.hpp"
    )
endif()
