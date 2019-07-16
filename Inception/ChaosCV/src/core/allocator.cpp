#include "core/allocator.hpp"

namespace chaos
{
	Allocator::~Allocator() {}

	PoolAllocator::PoolAllocator()
	{
		size_compare_ratio = 192;// 0.75f * 256
	}

	PoolAllocator::~PoolAllocator()
	{
		Clear();

		if (!payouts.empty())
		{
			LOG(ERROR) << "Pool allocator destroyed too early";

			std::list<std::pair<size_t, void*>>::iterator it = payouts.begin();
			for (; it != payouts.end(); it++)
			{
				void* ptr = it->second;
				LOG(ERROR) << ptr << " still in use";
				//fprintf(stderr, "%p still in use\n", ptr);
			}

			LOG(FATAL) << "Pool allocator destroyed too early";
		}
	}

	void PoolAllocator::Clear()
	{
		budgets_lock.lock();

		std::list< std::pair<size_t, void*> >::iterator it = budgets.begin();
		for (; it != budgets.end(); it++)
		{
			void* ptr = it->second;
			chaos::FastFree(ptr);
		}
		budgets.clear();

		budgets_lock.unlock();
	}

	void PoolAllocator::SetSizeCompareRatio(float scr)
	{
		CHECK(0.f < scr && scr < 1.f) << "Invalid size compare ratio " << scr;
		size_compare_ratio = (unsigned int)(scr * 256);
	}

	void* PoolAllocator::FastMalloc(size_t size)
	{
		budgets_lock.lock();

		// find free budget
		std::list<std::pair<size_t, void*>>::iterator it = budgets.begin();
		for (; it != budgets.end(); it++)
		{
			size_t bs = it->first;

			// size_compare_ratio ~ 100%
			if (bs >= size && ((bs * size_compare_ratio) >> 8) <= size)
			{
				void* ptr = it->second;
				budgets.erase(it);
				budgets_lock.unlock();
				payouts_lock.lock();
				payouts.push_back(std::make_pair(bs, ptr));
				payouts_lock.unlock();
				return ptr;
			}
		}
		budgets_lock.unlock();

		// new
		void* ptr = chaos::FastMalloc(size);
		payouts_lock.lock();
		payouts.push_back(std::make_pair(size, ptr));
		payouts_lock.unlock();
		return ptr;
	}

	void PoolAllocator::FastFree(void* ptr)
	{
		payouts_lock.lock();

		// return to budgets
		std::list< std::pair<size_t, void*>>::iterator it = payouts.begin();
		for (; it != payouts.end(); it++)
		{
			if (it->second == ptr)
			{
				size_t size = it->first;
				payouts.erase(it);
				payouts_lock.unlock();
				budgets_lock.lock();
				budgets.push_back(std::make_pair(size, ptr));
				budgets_lock.unlock();
				return;
			}
		}
		payouts_lock.unlock();

		LOG(ERROR) << "Pool allocator get wild " << ptr;
		chaos::FastFree(ptr);
		LOG(FATAL) << "Pool allocator get wild " << ptr;
	}


	UnlockedPoolAllocator::UnlockedPoolAllocator()
	{
		size_compare_ratio = 192;// 0.75f * 256
	}

	UnlockedPoolAllocator::~UnlockedPoolAllocator()
	{
		Clear();

		if (!payouts.empty())
		{
			LOG(ERROR) << "Unlocked pool allocator destroyed too early";

			std::list< std::pair<size_t, void*> >::iterator it = payouts.begin();
			for (; it != payouts.end(); it++)
			{
				void* ptr = it->second;
				LOG(ERROR) << ptr << " still in use";
			}
			LOG(FATAL) << "Pool allocator destroyed too early";
		}
	}

	void UnlockedPoolAllocator::Clear()
	{
		std::list< std::pair<size_t, void*> >::iterator it = budgets.begin();
		for (; it != budgets.end(); it++)
		{
			void* ptr = it->second;
			chaos::FastFree(ptr);
		}
		budgets.clear();
	}

	void UnlockedPoolAllocator::SetSizeCompareRatio(float scr)
	{
		CHECK(0.f < scr && scr < 1.f) << "Invalid size compare ratio " << scr;
		size_compare_ratio = (unsigned int)(scr * 256);

	}

	void* UnlockedPoolAllocator::FastMalloc(size_t size)
	{
		// find free budget
		std::list<std::pair<size_t, void*>>::iterator it = budgets.begin();
		for (; it != budgets.end(); it++)
		{
			size_t bs = it->first;

			// size_compare_ratio ~ 100%
			if (bs >= size && ((bs * size_compare_ratio) >> 8) <= size)
			{
				void* ptr = it->second;
				budgets.erase(it);
				payouts.push_back(std::make_pair(bs, ptr));
				return ptr;
			}
		}

		// new
		void* ptr = chaos::FastMalloc(size);
		payouts.push_back(std::make_pair(size, ptr));
		return ptr;
	}

	void UnlockedPoolAllocator::FastFree(void* ptr)
	{
		// return to budgets
		std::list<std::pair<size_t, void*>>::iterator it = payouts.begin();
		for (; it != payouts.end(); it++)
		{
			if (it->second == ptr)
			{
				size_t size = it->first;
				payouts.erase(it);
				budgets.push_back(std::make_pair(size, ptr));
				return;
			}
		}

		LOG(ERROR) << "Pool allocator get wild " << ptr;
		chaos::FastFree(ptr);
		LOG(FATAL) << "Pool allocator get wild " << ptr;
	}
}