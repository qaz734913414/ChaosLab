#pragma once

#include "core/core.hpp"
#include "core/allocator.hpp"

namespace chaos
{
	namespace dnn
	{
		class CHAOS_API Shape
		{
		public:
			Shape();

			template<class Type>
			Shape(const std::vector<Type>& data)
			{
				for (auto val : data)
				{
					shape.push_back((int)val);
				}
			}

			template<class Type>
			Shape(const std::initializer_list<Type>& list)
			{
				for (auto val : list)
				{
					shape.push_back((int)val);
				}
			}

			template<class Type>
			operator std::vector<Type>() const
			{
				std::vector<Type> ret;
				for (auto val : shape)
				{
					ret.push_back((Type)val);
				}
				return ret;
			}

			void Swap(Shape& shape);
			size_t Size() const;
			int operator[](size_t idx) const;

			CHAOS_API friend inline bool operator==(const Shape& s1, const Shape& s2)
			{
				return s1.shape == s2.shape;
			}
			CHAOS_API friend inline std::ostream& operator<<(std::ostream& stream, const Shape& shape);

			std::vector<int>::const_iterator begin() const;
			std::vector<int>::iterator begin();
			std::vector<int>::const_iterator end() const;
			std::vector<int>::iterator end();
			int& back();
			const int& back() const;

		private:
			std::vector<int> shape;
		};

		enum Depth
		{
			F32 = 4,
			F16 = 2,
			U8 = 1,
			UNK = -1,
		};

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

			Tensor Unroll(const std::vector<Mat>& vdata, bool rechannel = false, bool aligned = false, Allocator* allocator = nullptr);
			std::vector<Mat> Rollup(bool rechannel = false) const;

			/// <summary>
			/// <para>Create a tensor</para>
			/// <para>The buffer size is aligned to 4 bytes</para>
			/// </summary>
			void Create(const Shape& shape, const Depth& depth, bool aligned, Allocator* allocator);

			void Release();

			size_t Total() const;
			size_t Size() const;
			bool IsContinue() const;

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