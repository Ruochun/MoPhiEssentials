#ifndef MOPHI_ALGORITHMS_LINEAR_ALGEBRA_H
#define MOPHI_ALGORITHMS_LINEAR_ALGEBRA_H

#include "common/types.h"
#include "core/vector.h"

namespace MoPhi {
namespace Algorithms {

/**
 * @brief Basic linear algebra operations
 * 
 * This is a placeholder. Actual implementations should be copied from MoPhi.
 */

// Vector operations
template<typename T>
T dot(const Core::Vector<T>& a, const Core::Vector<T>& b);

template<typename T>
void axpy(T alpha, const Core::Vector<T>& x, Core::Vector<T>& y);

template<typename T>
T norm(const Core::Vector<T>& v);

} // namespace Algorithms
} // namespace MoPhi

#endif // MOPHI_ALGORITHMS_LINEAR_ALGEBRA_H
