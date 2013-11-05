/** Simple Spin Lock -*- C++ -*-
 * @file
 * @section License
 *
 * Galois, a framework to exploit amorphous data-parallelism in
 * irregular programs.
 *
 * Copyright (C) 2013, The University of Texas at Austin. All rights
 * reserved.  UNIVERSITY EXPRESSLY DISCLAIMS ANY AND ALL WARRANTIES
 * CONCERNING THIS SOFTWARE AND DOCUMENTATION, INCLUDING ANY
 * WARRANTIES OF MERCHANTABILITY, FITNESS FOR ANY PARTICULAR PURPOSE,
 * NON-INFRINGEMENT AND WARRANTIES OF PERFORMANCE, AND ANY WARRANTY
 * THAT MIGHT OTHERWISE ARISE FROM COURSE OF DEALING OR USAGE OF
 * TRADE.  NO WARRANTY IS EITHER EXPRESS OR IMPLIED WITH RESPECT TO
 * THE USE OF THE SOFTWARE OR DOCUMENTATION. Under no circumstances
 * shall University be liable for incidental, special, indirect,
 * direct or consequential damages or loss of profits, interruption of
 * business, or related expenses which may arise from use of Software
 * or Documentation, including but not limited to those resulting from
 * defects in Software and/or Documentation, or loss or inaccuracy of
 * data of any kind.  
 *
 * @section Description
 *
 * This contains the basic spinlock used in Galois.  We use a
 * test-and-test-and-set approach, with pause instructions on x86.
 * This implements C++11 lockable and try lockable concept
 *
 * @author Andrew Lenharth <andrew@lenharth.org>
 */
#ifndef GALOIS_RUNTIME_LL_SIMPLE_LOCK_H
#define GALOIS_RUNTIME_LL_SIMPLE_LOCK_H

#include "Galois/config.h"
#include "Galois/Runtime/ll/CompilerSpecific.h"

#include <cassert>
#include GALOIS_CXX11_STD_HEADER(atomic)

namespace Galois {
namespace Runtime {
namespace LL {

/// SimpleLock is a spinlock.
/// Copying a lock is unsynchronized (relaxed ordering)

class SimpleLock {
  mutable std::atomic<int> _lock;
  void slow_lock() const;

public:
  constexpr SimpleLock() : _lock(0) {  }
  //relaxed order for copy
  SimpleLock(const SimpleLock& p) :_lock(p._lock.load(std::memory_order_relaxed)) {}

  SimpleLock& operator= (const SimpleLock& p) {
    if (&p == this) return *this;
    //relaxed order for initialization
    _lock.store(p._lock.load(std::memory_order_relaxed), std::memory_order_relaxed);
    return *this;
  }

  inline void lock() const {
    int oldval = 0;
    if (_lock.load(std::memory_order_relaxed))
      goto slow_path;
    if (!_lock.compare_exchange_weak(oldval, 1, std::memory_order_acq_rel, std::memory_order_relaxed))
      goto slow_path;
    assert(is_locked());
    return;
  slow_path:
    slow_lock();
  }

  inline void unlock() const {
    assert(is_locked());
    //HMMMM
    _lock.store(0, std::memory_order_release);
    //_lock = 0;
  }

  inline bool try_lock() const {
    int oldval = 0;
    if (_lock.load(std::memory_order_acquire))
      return false;
    if (!_lock.compare_exchange_weak(oldval, 1, std::memory_order_acq_rel))
      return false;
    assert(is_locked());
    return true;
  }

  inline bool is_locked() const {
    return _lock.load(std::memory_order_acquire) & 1;
  }
};

  //Dummy Lock implements the lock interface without a lock

class DummyLock {
public:
  inline void lock() const {}
  inline void unlock() const {}
  //  inline bool try_lock() const { return true; }
  //inline bool is_locked() const { }
};

}
}
} // end namespace Galois

#endif
