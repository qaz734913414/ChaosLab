#pragma once

#include "test/test_data.hpp"

namespace chaos
{
	namespace test
	{
		class CHAOS_API TestEngine
		{
		public:
			virtual void Run() = 0;
			virtual void Report() = 0;
			virtual void Close() = 0;
		};

		/// <summary>
		/// <para>Identification Test</para>
		/// <para></para>
		/// </summary>
		class CHAOS_API ITest : public TestEngine
		{
		public:

		protected:

		};
	}
}