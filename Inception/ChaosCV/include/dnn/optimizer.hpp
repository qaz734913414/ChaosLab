#pragma once

#include "dnn/tensor.hpp"
#include "dnn/net.hpp"

namespace chaos
{
	namespace dnn
	{
		class CHAOS_API Optimizer
		{
		public:
			//virtual void Optimize() = 0;
			virtual ~Optimizer() {}

			virtual void MergeBatchNorm() = 0;

			virtual void Export(const std::string& name) = 0;

			static Ptr<Optimizer> LoadMxNet(const Model& model);
		};
	}
}