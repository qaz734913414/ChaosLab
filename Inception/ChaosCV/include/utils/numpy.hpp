#pragma once

#include "core/core.hpp"
#include "dnn/tensor.hpp"

namespace chaos
{
	class CHAOS_API Numpy
	{
	public:
		Numpy(const File& file);

		void CreateHead(const Shape& shape, const Depth& depth);

		void Add(const dnn::Tensor& tensor);

		static dnn::Tensor Load(const File& file);

	private:
		Numpy(const Numpy& npy) = delete;
		Numpy(Numpy&& npy) = delete;

		File file;
	};
}