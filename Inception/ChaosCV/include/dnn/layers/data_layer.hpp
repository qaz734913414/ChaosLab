#pragma once

#include "core/core.hpp"
#include "dnn/tensor.hpp"

namespace chaos
{
	namespace dnn
	{
		class CHAOS_API DataLayer
		{
		public:
			DataLayer();
			DataLayer(const std::string& name, const Shape& shape);

			std::string name;
			Shape shape;
		};
	} // namespace chaos
} // namespace chaos