//  Copyright (c) 2025, Ruochun Zhang
//	SPDX-License-Identifier: BSD-3-Clause

#ifndef MOPHI_ESSENTIALS_APIVER_H
#define MOPHI_ESSENTIALS_APIVER_H

// Project Version Macros
// These can be overridden by CMake configuration if needed
#ifndef MOPHI_ESSENTIALS_VERSION_MAJOR
#define MOPHI_ESSENTIALS_VERSION_MAJOR 1
#endif

#ifndef MOPHI_ESSENTIALS_VERSION_MINOR
#define MOPHI_ESSENTIALS_VERSION_MINOR 0
#endif

// The project version number expressed in the form 0xMMMMmmPP (for easy numerical comparisons)
#define MOPHI_ESSENTIALS_API_VERSION ((MOPHI_ESSENTIALS_VERSION_MAJOR << 16) | (MOPHI_ESSENTIALS_VERSION_MINOR << 8))

// C++ Standard Macros
#define STD_AUTODETECT (__cplusplus)
#define STD_CXX98 199711L
#define STD_CXX11 201103L
#define STD_CXX14 201402L
#define STD_CXX17 201703L
#define STD_CXX20 202002L

// The C++ Standard Version targeted by the library
// Defaults to auto-detect (use actual __cplusplus value)
// Can be overridden by CMake configuration
#ifndef CXX_TARGET
#define CXX_TARGET STD_AUTODETECT
#endif

// C++ Standard Comparisons
#define CXX_EQUAL(x)    (CXX_TARGET == x)
#define CXX_NEWER(x)    (CXX_TARGET >  x)
#define CXX_OLDER(x)    (CXX_TARGET <  x)

// C++ Standard Composite Comparisons
#define CXX_EQ_NEWER(x)	(CXX_EQUAL(x) || CXX_NEWER(x))
#define CXX_EQ_OLDER(x) (CXX_EQUAL(x) || CXX_OLDER(x))

// CUDA Toolkit Headers path (if needed, can be set by CMake)
// This is typically not used in code, mainly for documentation/debugging
#ifndef MOPHI_ESSENTIALS_CUDA_TOOLKIT_HEADERS
#define MOPHI_ESSENTIALS_CUDA_TOOLKIT_HEADERS ""
#endif

#endif
