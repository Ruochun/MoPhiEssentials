//  Copyright (c) 2025, Ruochun Zhang
//	SPDX-License-Identifier: BSD-3-Clause

#ifndef MOPHI_ESSENTIALS_APIVER_H
#define MOPHI_ESSENTIALS_APIVER_H

// C++ Standard Macros
#define STD_AUTODETECT (__cplusplus)
#define STD_CXX98 199711L
#define STD_CXX11 201103L
#define STD_CXX14 201402L
#define STD_CXX17 201703L
#define STD_CXX20 202002L

// C++ Standard Comparisons
#define CXX_EQUAL(x) (STD_AUTODETECT == x)
#define CXX_NEWER(x) (STD_AUTODETECT > x)
#define CXX_OLDER(x) (STD_AUTODETECT < x)

// C++ Standard Composite Comparisons
#define CXX_EQ_NEWER(x) (CXX_EQUAL(x) || CXX_NEWER(x))
#define CXX_EQ_OLDER(x) (CXX_EQUAL(x) || CXX_OLDER(x))

#endif
