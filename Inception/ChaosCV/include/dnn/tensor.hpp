#pragma once

#include "core/core.hpp"
#include "core/allocator.hpp"

namespace chaos
{
	namespace dnn
	{

		/// <summary>
		/// <para>Tensor</para>
		/// <para>Refer to OpenCV cv::Mat and NCNN ncnn::Mat</para>
		/// </summary>
		class CHAOS_API Tensor
		{
		public:
			Tensor();
			Tensor(const Shape& shape, const Depth& depth, bool aligned = false, Allocator* allocator = nullptr);
			Tensor(const Shape& shape, const Depth& depth, void* data, bool aligned = false, Allocator* allocator = nullptr);

			Tensor(const Tensor& tensor);

			~Tensor();

			Tensor& operator=(const Tensor& tensor);

			static Tensor Unroll(const std::vector<Mat>& vdata, bool rechannel = false, bool aligned = false, Allocator* allocator = nullptr);
			std::vector<Mat> Rollup(bool rechannel = false) const;

			// Return a Mat which data pointer is to Tensor.data
			operator Mat() const;

			/// <summary>
			/// <para>Create a tensor</para>
			/// <para>The buffer size is aligned to 4 bytes</para>
			/// </summary>
			void Create(const Shape& shape, const Depth& depth, bool aligned, Allocator* allocator);

			void Release();

			template<class Type>
			inline Type At(const std::vector<int>& position) const
			{
				CHECK_EQ(dims, position.size());
				for (int i = 0; i < dims; i++)
				{
					CHECK_LT(position[i], shape[i]);
				}

				std::vector<int> steps(dims, 1);
				if (dims > 1) steps[dims - 2LL] = shape.back();
				if (dims > 2) steps[dims - 3LL] = (int)cstep;
				if (dims > 3)
				{
					for (int i = dims - 4; i >= 0; i--)
					{
						steps[i] = steps[i + 1LL] * shape[i + 1LL];
					}
				}

				size_t offset = 0;
				for (int i = 0; i < dims; i++)
				{
					offset += (size_t)steps[i] * position[i];
				}
				return *((Type*)data + offset);
			}

			Tensor Flatten() const;
			Tensor Reshape(const Shape& new_shape);

			size_t Total() const;
			size_t Size() const;
			bool IsContinue() const;

			/// <summary>
			/// <para>Convert Tensor to Mat and fill to ostream</para>
			/// </summary>
			CHAOS_API friend inline std::ostream& operator<<(std::ostream& stream, const Tensor& tensor);

			void* data; // pointer to the data
			Allocator* allocator; // the allocator

			// pointer to the reference counter
			// when points to user-allocated data, the pointer is NULL
			int* ref_cnt;

			Shape shape;
			int dims; // dims;

			Depth depth;

			bool aligned;
			size_t cstep; // channel step;
		};
	}
}