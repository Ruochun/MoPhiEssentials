#ifndef MOPHI_ALGORITHMS_SORTING_H
#define MOPHI_ALGORITHMS_SORTING_H

#include "common/types.h"
#include "core/vector.h"

namespace MoPhi {
namespace Algorithms {

/**
 * @brief Sorting algorithms for various data types
 * 
 * This is a placeholder. Actual implementations should be copied from MoPhi.
 */
template<typename T>
void sort(Core::Vector<T>& data);

template<typename T, typename Compare>
void sort(Core::Vector<T>& data, Compare comp);

} // namespace Algorithms
} // namespace MoPhi

#endif // MOPHI_ALGORITHMS_SORTING_H
