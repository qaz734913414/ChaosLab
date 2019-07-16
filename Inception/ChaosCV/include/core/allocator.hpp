#pragma once

#pragma once

#include "core/core.hpp"

#include <list>
#include <mutex>

#include <intrin.h>

#define MALLOC_ALIGN 16

// exchange-add operation for atomic operations on reference counters
// Just for windows, reference to NCNN
#define CHAOS_XADD(addr, delta) (int)_InterlockedExchangeAdd((long volatile*)addr, delta)

namespace chaos
{
	/// <summary>
	/// <para>Aligns a pointer to the specified number of bytes</para>
	/// <para>The function returns the aligned pointer of the same type as the input pointer:</para>
	/// <para>(_Tp*)(((size_t)ptr + n - 1) and -n)</para>
	/// </summary>
	/// <param name="ptr">Aligned pointer</param>
	/// <param name="n">Alignment size that must be a power of two</param>
	/// <return>The aligned pointer of the same type as the input pointer</return>
	template<typename _Tp> static inline _Tp* AlignPtr(_Tp* ptr, int n = (int)sizeof(_Tp))
	{
		CHECK((n & (n - 1)) == 0); // n is a power of 2
		return (_Tp*)(((size_t)ptr + n - 1) & -n);
	}

	/// <summary>
	/// <para>Aligns a buffer size to the specified number of bytes</para>
	/// <para>The function returns the minimum number that is greater than or equal to sz and is divisible by n :</para>
	/// <para>(sz + n - 1) and -n</para>
	/// </summary>
	/// <param name="sz">Buffer size to align</param>
	/// <param name="n">Alignment size that must be a power of two</param>
	/// <return>The minimum number that is greater than or equal to sz and is divisible by n</return>
	static inline size_t AlignSize(size_t sz, int n)
	{
		CHECK((n & (n - 1)) == 0); // n is a power of 2
		return (sz + n - 1) & -n;
	}

	//static void* OutOfMemoryError(size_t size)
	//{
	//	LOG(FATAL) << "Failed to allocate " << size << " bytes";
	//}

	static inline void* FastMalloc(size_t size)
	{
		return _aligned_malloc(size, MALLOC_ALIGN);
	}

	static inline void FastFree(void* ptr)
	{
		if (ptr)
		{
			_aligned_free(ptr);
		}
	}

	class Allocator
	{
	public:
		virtual ~Allocator();

		virtual void* FastMalloc(size_t size) = 0;
		virtual void FastFree(void* ptr) = 0;
	};


	class PoolAllocator : public Allocator
	{
	public:
		PoolAllocator();
		~PoolAllocator();

		// ratio range 0 ~ 1
		// default cr = 0.75
		void SetSizeCompareRatio(float scr);

		// release all budgets immediately
		void Clear();

		virtual void* FastMalloc(size_t size);
		virtual void FastFree(void* ptr);

	private:
		std::mutex budgets_lock;
		std::mutex payouts_lock;

		unsigned int size_compare_ratio;// 0~256

		std::list<std::pair<size_t, void*>> budgets;
		std::list<std::pair<size_t, void*>> payouts;
	};

	class UnlockedPoolAllocator : public Allocator
	{
	public:
		UnlockedPoolAllocator();
		~UnlockedPoolAllocator();

		// ratio range 0 ~ 1
		// default cr = 0.75
		void SetSizeCompareRatio(float scr);

		// release all budgets immediately
		void Clear();

		virtual void* FastMalloc(size_t size);
		virtual void FastFree(void* ptr);

	protected:
		unsigned int size_compare_ratio;// 0~256

		std::list<std::pair<size_t, void*>> budgets;
		std::list<std::pair<size_t, void*>> payouts;
	};
}