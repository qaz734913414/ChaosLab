#include "dnn/tensor.hpp"

namespace chaos
{
	namespace dnn
	{
		inline int Cast(const Depth& depth)
		{
			switch (depth)
			{
			case F32:
				return CV_32F;
			case F16:
				return CV_16F;
			case U8:
				return CV_8U;
			default:
				LOG(FATAL) << "Unknown depth";
				return -1;
			}
		}

		inline Depth Cast(int depth)
		{
			switch (depth)
			{
			case CV_32F:
				return F32;
			case CV_16F:
				return F16;
			case CV_8U:
				return U8;
			default:
				LOG(FATAL) << "Unknown depth";
				return UNK;
			}
		}

		Shape::Shape() : shape(std::vector<int>()) {}

		void Shape::Swap(Shape& _shape)
		{
			shape.swap(_shape.shape);
		}
		size_t Shape::Size() const { return shape.size(); }
		int Shape::operator[](size_t idx) const { return shape[idx]; }
		inline std::ostream& operator<<(std::ostream& stream, const Shape& shape)
		{
			stream << "[";
			for (int i = 0; i < shape.Size() - 1; i++)
			{
				stream << shape[i] << ", ";
			}
			stream << shape[shape.Size() - 1] << "]";
			return stream;
		}

		std::vector<int>::const_iterator Shape::begin() const
		{
			return shape.begin();
		}
		std::vector<int>::iterator Shape::begin()
		{
			return shape.begin();
		}
		std::vector<int>::const_iterator Shape::end() const
		{
			return shape.end();
		}
		std::vector<int>::iterator Shape::end()
		{
			return shape.end();
		}
		const int& Shape::back() const
		{
			return shape.back();
		}
		int& Shape::back()
		{
			return shape.back();
		}






		Tensor::Tensor() : data(nullptr), ref_cnt(nullptr), allocator(nullptr), depth(UNK), shape(Shape()), cstep(0), dims(0), aligned(false) {}

		Tensor::Tensor(const Shape& shape, const Depth& depth, bool aligned, Allocator* allocator) : Tensor()
		{
			Create(shape, depth, aligned, allocator);
		}
		Tensor::Tensor(const Shape& shape, const Depth& depth, void* data, bool aligned, Allocator* allocator) 
			: data(data), ref_cnt(nullptr), aligned(aligned), shape(shape), dims(static_cast<int>(shape.Size())), depth(depth), allocator(allocator)
		{
			CHECK_NE(UNK, depth);
			cstep = dims >= 2 ? (size_t)shape[dims - 1LL] * shape[dims - 2LL] : shape[0];
			if (aligned) cstep = AlignSize(cstep * depth, MALLOC_ALIGN) / depth;
		}

		Tensor::Tensor(const Tensor& t) 
			: data(t.data), ref_cnt(t.ref_cnt), aligned(t.aligned), shape(t.shape), dims(t.dims), depth(t.depth), cstep(t.cstep), allocator(t.allocator)
		{
			if (ref_cnt)
				CHAOS_XADD(ref_cnt, 1);
		}

		Tensor::~Tensor()
		{
			Release();
		}

		Tensor& Tensor::operator=(const Tensor& t)
		{
			if (this == &t)
				return *this;

			if (t.ref_cnt)
				CHAOS_XADD(t.ref_cnt, 1);

			Release();

			data = t.data;
			ref_cnt = t.ref_cnt;
			aligned = t.aligned;
			shape = t.shape;
			dims = t.dims;
			depth = t.depth;
			cstep = t.cstep;
			allocator = t.allocator;

			return *this;
		}

		Tensor Tensor::Unroll(const std::vector<Mat>& vdata, bool rechannel, bool aligned, Allocator* allocator)
		{
			CHECK(!vdata.empty());

			Shape shape = { (int)vdata.size(), vdata[0].channels(), vdata[0].rows, vdata[0].cols };
			Tensor tensor = Tensor(shape, Cast(vdata[0].depth()), aligned, allocator);

			for (int n = 0; n < tensor.shape[0]; n++)
			{
				CHECK_EQ(2, vdata[n].dims);
				CHECK_EQ(vdata[0].depth(), vdata[n].depth());
				CHECK_EQ(shape[1], vdata[n].channels());
				CHECK_EQ(shape[2], vdata[n].rows);
				CHECK_EQ(shape[3], vdata[n].cols);

				std::vector<Mat> slice(tensor.shape[1]);
				int c = rechannel ? shape[1] - 1 : 0;
				for (int i = 0; i < shape[1]; i++)
				{
					int idx = std::abs(c - i);
					slice[idx] = Mat(vdata[n].size(), vdata[n].depth(), (uchar*)tensor.data + ((size_t)shape[1] * n + i) * tensor.cstep * vdata[n].elemSize1());
				}
				cv::split(vdata[n], slice);
			}

			return tensor;
		}

		std::vector<Mat> Tensor::Rollup(bool rechannel) const
		{
			CHECK_EQ(4, dims);

			std::vector<Mat> _data;
			for (int n = 0; n < shape[0]; n++)
			{
				Mat packed;
				std::vector<cv::Mat> slice(shape[1]);
				int c = rechannel ? shape[1] - 1 : 0;
				for (int i = 0; i < shape[1]; i++)
				{
					int idx = std::abs(c - i);
					slice[idx] = cv::Mat(shape[2], shape[3], Cast(depth), (uchar*)data + ((size_t)shape[1] * n + i) * cstep * depth);
				}
				cv::merge(slice, packed);

				_data.push_back(packed);
			}

			return _data;
		}





		void Tensor::Create(const Shape& _shape, const Depth& _depth, bool _aligned, Allocator* _allocator)
		{
			if (shape == _shape && depth == _depth && allocator == _allocator) return;

			Release();

			CHECK_NE(UNK, _depth);

			shape = _shape;
			depth = _depth;
			aligned = _aligned;
			allocator = _allocator;

			dims = static_cast<int>(shape.Size());

			cstep = dims > 1 ? (size_t)shape[dims - 1LL] * shape[dims - 2LL] : shape[0];
			if (aligned) cstep = AlignSize(cstep * depth, MALLOC_ALIGN) / depth;

			if (Total() > 0)
			{
				size_t size = AlignSize(Total() * depth, 4);
				if (allocator)
					data = allocator->FastMalloc(size + sizeof(*ref_cnt));
				else
					data = FastMalloc(size + sizeof(*ref_cnt));
				ref_cnt = (int*)(((uchar*)data) + size);
				*ref_cnt = 1;
			}
		}


		void Tensor::Release()
		{
			if (ref_cnt && CHAOS_XADD(ref_cnt, -1) == 1)
			{
				if (allocator)
					allocator->FastFree(data);
				else
					FastFree(data);
			}

			Shape().Swap(shape);
			depth = UNK;
			dims = 0;
			aligned = false;

			data = nullptr;
			ref_cnt = nullptr;
		}


		Tensor Tensor::Reshape(const Shape& new_shape)
		{
			size_t new_size = 1;
			for (auto n : new_shape)
			{
				new_size *= n;
			}
			CHECK_EQ(Size(), new_size) << "Must keep the number of elements same.";

			Tensor new_tensor = Tensor(new_shape, depth, aligned, allocator);

			if (!aligned)
			{
				memcpy(new_tensor.data, data, Size() * depth);
			}
			else
			{
				LOG(FATAL) << "Now can not reshape the matrix which is aligned.";
			}

			return new_tensor;
		}

		size_t Tensor::Total() const
		{
			size_t total = cstep;
			for (int i = 0; i < dims - 2; i++) total *= shape[i];
			return total;
		}

		size_t Tensor::Size() const
		{
			size_t size = 1;
			for (auto val : shape) size *= val;
			return size;
		}

		bool Tensor::IsContinue() const
		{
			if (dims < 3) return true;
			return cstep == (size_t)shape[dims - 1LL] * shape[dims - 2LL];
		}

		inline std::ostream& operator<<(std::ostream& stream, const Tensor& tensor)
		{
			if (tensor.dims == 1)
			{
				stream << Mat(1, tensor.shape[0], Cast(tensor.depth), tensor.data) << std::endl;
			}
			else
			{
				int num = 1;
				for (int i = 0; i < tensor.dims - 2; i++)
				{
					num *= tensor.shape[i];
				}

				int h = tensor.shape[tensor.dims - 2LL];
				int w = tensor.shape[tensor.dims - 1LL];
				char* slice = (char*)tensor.data;
				for (int i = 0; i < num - 1; i++)
				{
					stream << Mat(h, w, Cast(tensor.depth), slice) << std::endl;
					slice += tensor.cstep * tensor.depth;
				}
				stream << Mat(h, w, Cast(tensor.depth), slice);
			}
			return stream;
		}
	}
}