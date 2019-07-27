#pragma once

#include "core/core.hpp"

namespace chaos
{
	namespace test
	{
		class CHAOS_API CumulativeTabel
		{
		public:
			CumulativeTabel();
			CumulativeTabel(int noc);

			void Apply(int actual_id, const std::vector<double>& prob);

			Mat GetPrecision() const;

			CumulativeTabel& operator+(const CumulativeTabel& c);
		private:
			Mat table; // 1 x NOC;
		};
	}
}